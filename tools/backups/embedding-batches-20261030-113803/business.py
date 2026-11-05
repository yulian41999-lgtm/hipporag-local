"""Prepare banking corpus, build an index, and run gold-free business QA."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from settings import ROOT, load_settings
from business_qa import answer_question


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding="utf-8")


def create_rag(cfg, options, index_dir):
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig
    from local_embedding import LocalBGEEmbedding
    config = BaseConfig(
        save_dir=str(index_dir), llm_name=cfg["llm_model"], llm_base_url=cfg["llm_base_url"],
        llm_api_key_env=cfg["llm_api_key_env"], embedding_model_name=cfg["embedding_model_id"],
        embedding_provider="transformers", embedding_batch_size=cfg["batch_size"],
        embedding_max_seq_len=cfg["max_seq_length"], embedding_model_dtype="auto",
        embedding_return_as_normalized=True, openie_mode="online",
        openie_max_workers=int(os.environ.get("HIPPO_OPENIE_MAX_WORKERS", cfg.get("openie_max_workers", 1))),
        max_retry_attempts=1, retrieval_top_k=options["retrieval_top_k"], qa_top_k=options["qa_top_k"],
        max_qa_steps=1, linking_top_k=5,
    )
    embedding = LocalBGEEmbedding(config, cfg)
    identity = {
        "adapter": "local-bge-dense-v1", "model": cfg["embedding_model_id"],
        "revision": cfg["embedding_revision"], "max_seq_length": cfg["max_seq_length"],
        "normalized": True, "dtype": "float16" if embedding.device.startswith("cuda") else "float32",
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    try:
        return HippoRAG(global_config=config, embedding_model=embedding, index_identity=digest), embedding
    except Exception:
        embedding.close()
        raise


def split_chunks(text, title, tokenizer, max_tokens, overlap):
    """Keep original character spans and account for the title and special tokens."""
    title_ids = tokenizer(title, add_special_tokens=False)["input_ids"]
    header = (title if len(title_ids) <= 96 else tokenizer.decode(title_ids[:96])) + "\n"
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)["offset_mapping"]
    budget = max_tokens - len(tokenizer(header)["input_ids"]) - 4
    if not 0 <= overlap < budget:
        raise ValueError("Chunk overlap must be smaller than body token budget")
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        while end > start:
            first, last = offsets[start][0], offsets[end - 1][1]
            content = header + text[first:last]
            if len(tokenizer(content, truncation=False)["input_ids"]) <= max_tokens:
                break
            end -= 1
        if end <= start:
            raise ValueError("Cannot fit a source span into the embedding limit")
        yield content, first, last
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)


def prepare(cfg, options, limit=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["embedding_path"], local_files_only=True)
    parser_path = ROOT / "data/banking/银行监管测试集/tools/inspect_corpus.py"
    spec = importlib.util.spec_from_file_location("banking_raw_parser", parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    raw_dir = ROOT / options["raw_dir"]
    grouped = {}
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".html", ".htm", ".txt"}:
            grouped.setdefault(path.stem.split("__", 1)[0], []).append(path)
    selected = []
    for doc_id, paths in sorted(grouped.items()):
        preferred = [p for p in paths if p.suffix.lower() in {".html", ".htm"}] or paths
        if len(preferred) != 1:
            raise ValueError("Ambiguous source files for document: " + doc_id)
        selected.append((doc_id, preferred[0]))
    if limit:
        selected = selected[:limit]
    chunks = []
    skipped = []
    for doc_id, path in selected:
        if path.suffix.lower() in {".html", ".htm"}:
            parsed = parser.load_doc(path)
            text, title = "\n".join(parsed["lines"]), parsed["title"]
            metadata = {k: v for k, v in parsed.items() if k != "lines"}
        else:
            text = path.read_text(encoding="utf-8-sig").replace("\x00", "\ufffd").strip()
            title = path.stem.split("__", 1)[-1]
            metadata = {"doc_id": doc_id, "title": title, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if not text.strip():
            skipped.append(doc_id)
            continue
        for i, (content, first, last) in enumerate(split_chunks(text, title, tokenizer, cfg["max_seq_length"], options["chunk_overlap_tokens"])):
            chunks.append({
                "text": content, "source_id": doc_id,
                "metadata": {**metadata, "source_file": path.relative_to(ROOT).as_posix(), "chunk_id": f"{doc_id}:{i}", "char_start": first, "char_end": last},
            })
    if not chunks:
        raise ValueError("No corpus chunks were prepared")
    corpus_path = ROOT / options["corpus_file"]
    if limit:
        corpus_path = corpus_path.with_name("corpus.preview.json")
    dump(corpus_path, chunks)
    report = {"scope": "preview" if limit else "full_corpus_preparation", "raw_file_count": sum(len(p) for p in grouped.values()), "document_count": len(selected), "chunk_count": len(chunks), "skipped_empty_documents": skipped, "duplicate_format_policy": "prefer HTML over TXT with same doc_id", "embedding_max_seq_length": cfg["max_seq_length"], "overlap_tokens": options["chunk_overlap_tokens"], "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest()}
    dump(corpus_path.with_suffix(".manifest.json"), report)
    print(json.dumps(report, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "index", "answer", "smoke"])
    parser.add_argument("--limit-documents", type=int)
    args = parser.parse_args()
    cfg = load_settings()
    options = json.loads((ROOT / "business_config.json").read_text(encoding="utf-8"))
    if os.environ.get("HIPPO_OUTPUT_DIR"):
        base = Path(cfg["output_dir"]) / "banking"
        options.update(corpus_file=str(base / "corpus.json"), index_dir=str(base / "index"), predictions_file=str(base / "predictions.json"))
    if args.limit_documents is not None and (args.mode != "prepare" or args.limit_documents < 1):
        parser.error("--limit-documents is only for prepare previews, and must be positive")
    if args.mode == "prepare":
        prepare(cfg, options, args.limit_documents)
        return
    index_dir = ROOT / options["index_dir"]
    if args.mode == "smoke":
        index_dir = ROOT / "outputs/smoke_chinese_demo/index"
    if args.mode != "index" and not list(index_dir.rglob("graph.pickle")):
        raise FileNotFoundError("Index is missing; prepare the corpus and run index first")
    if args.mode == "answer":
        manifest_path = index_dir / "business_corpus_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Business index has no completed corpus manifest")
        state = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_hash = hashlib.sha256((ROOT / options["corpus_file"]).read_bytes()).hexdigest()
        if state.get("state") != "ready" or state["corpus_sha256"] != current_hash:
            raise ValueError("Business index is incomplete or the prepared corpus changed")
    rag, embedding = create_rag(cfg, options, index_dir)
    try:
        with rag:
            if args.mode == "index":
                from hipporag import Chunk
                corpus_path = ROOT / options["corpus_file"]
                corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
                fingerprint = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
                manifest_path = index_dir / "business_corpus_manifest.json"
                if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["corpus_sha256"] != fingerprint:
                    raise ValueError("Prepared corpus changed; select a fresh business index_dir")
                if not manifest_path.exists() and rag.graph.vcount():
                    raise ValueError("Existing graph has no corpus manifest; select a fresh business index_dir")
                dump(manifest_path, {"corpus_sha256": fingerprint, "chunks": len(corpus), "state": "building"})
                rag.index([Chunk(content=c["text"], source_id=c["source_id"], metadata=c["metadata"]) for c in corpus])
                dump(manifest_path, {"corpus_sha256": fingerprint, "chunks": len(corpus), "state": "ready"})
                print("Business corpus indexed.")
                return
            if args.mode == "smoke":
                questions = [{"id": "synthetic-business-qa-smoke-001", "question": "澄川银行总部所在的城市属于哪个省？"}]
                output = ROOT / "outputs/business_smoke/predictions.json"
            else:
                questions = json.loads((ROOT / options["questions_file"]).read_text(encoding="utf-8"))
                if len(questions) != 15 or len({q["id"] for q in questions}) != 15 or any(set(q) != {"id", "question"} for q in questions):
                    raise ValueError("Inference input must contain exactly 15 unique id/question records, with no gold fields")
                output = ROOT / options["predictions_file"]
            if output.exists():
                raise FileExistsError("Predictions already exist; change predictions_file or preserve the previous run first")
            predictions = []
            for question in questions:
                try:
                    result = answer_question(rag, question["question"], top_k=options["qa_top_k"], context_token_budget=options["context_bge_token_budget"])
                except Exception as exc:
                    result = {"question": question["question"], "generated_answer": "", "context": [], "status": "error", "error": type(exc).__name__}
                predictions.append({"id": question["id"], "generation_model": cfg["llm_model"], "retrieval_top_k": options["retrieval_top_k"], "qa_top_k": options["qa_top_k"], **result})
                dump(output, predictions)
                print(f"Answered {len(predictions)}/{len(questions)}", flush=True)
            if args.mode == "smoke":
                # Synthetic smoke reference, kept separate from formal question data.
                dump(output.with_name("eval_input.json"), [{
                    "id": predictions[0]["id"], "question": questions[0]["question"], "question_type": "Complex Reasoning",
                    "ground_truth": "澄川银行的总部位于海州市。海州市属于临江省。因此澄川银行总部所在城市属于临江省。",
                    "generated_answer": predictions[0]["generated_answer"], "context": predictions[0]["context"],
                    "evidence": ["澄川银行的总部位于海州市。", "海州市属于临江省。"],
                }])
            if any(row["status"] != "ok" for row in predictions):
                raise SystemExit(2)
    finally:
        embedding.close()


if __name__ == "__main__":
    main()
