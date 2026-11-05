"""Run local Embedding checks, official sample, and persisted-index checks."""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import logging
from pathlib import Path
import time
import numpy as np
from settings import ROOT, load_settings


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["embedding", "sample", "reload"])
    parser.add_argument("--config")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dataset", choices=["sample", "chinese_demo"], default="sample")
    args = parser.parse_args()
    cfg = load_settings(args.config)
    if args.device:
        cfg["device"] = args.device
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig
    from local_embedding import LocalBGEEmbedding
    import torch

    logging.basicConfig(level=logging.INFO)
    output = ROOT / cfg["output_dir"]
    if args.dataset != "sample":
        output = output.with_name(output.name + "_" + args.dataset)
    output.mkdir(parents=True, exist_ok=True)
    config = BaseConfig(
        save_dir=str(output / "index"),
        llm_name=cfg["llm_model"], llm_base_url=cfg["llm_base_url"],
        llm_api_key_env=cfg["llm_api_key_env"],
        embedding_model_name=cfg["embedding_model_id"],
        embedding_provider="transformers", embedding_batch_size=cfg["batch_size"],
        embedding_max_seq_len=cfg["max_seq_length"],
        embedding_model_dtype="auto", embedding_return_as_normalized=True,
        openie_mode="online", openie_max_workers=1,
        max_retry_attempts=1, retrieval_top_k=3, qa_top_k=3,
        max_qa_steps=1, linking_top_k=5,
    )
    started = time.monotonic()
    embedding = LocalBGEEmbedding(config, cfg)
    try:
        if args.mode == "embedding":
            texts = ["银行资本充足率监管要求。", "商业银行需要符合资本监管要求。", "今天的天气晴朗。"]
            vectors = embedding.batch_encode(texts)
            repeated = embedding.batch_encode(texts[:1])
            assert vectors.shape == (3, 1024), vectors.shape
            assert np.isfinite(vectors).all()
            assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            assert np.allclose(vectors[0], repeated[0], atol=1e-4)
            assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2], "Related text should rank above unrelated text."
            report = {
                "mode": args.mode, "device": embedding.device,
                "shape": list(vectors.shape), "dtype": str(vectors.dtype),
                "similarities_to_first": (vectors @ vectors[0]).tolist(),
                "gpu_peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if embedding.device.startswith("cuda") else None,
            }
        else:
            source = ROOT / "vendor" / "HippoRAG"
            dataset_dir = source / "reproduce/dataset" if args.dataset == "sample" else ROOT / "data"
            corpus = json.loads((dataset_dir / f"{args.dataset}_corpus.json").read_text(encoding="utf-8"))
            samples = json.loads((dataset_dir / f"{args.dataset}.json").read_text(encoding="utf-8"))
            docs = [f"{row['title']}\n{row['text']}" for row in corpus]
            questions = [row["question"] for row in samples]
            gold_docs = [[f"{p['title']}\n{p['text']}" for p in row["paragraphs"] if p.get("is_supporting", True)] for row in samples]
            identity = {
                "adapter": "local-bge-dense-v1",
                "model": cfg["embedding_model_id"], "revision": cfg["embedding_revision"],
                "max_seq_length": cfg["max_seq_length"], "normalized": True,
                "dtype": "float16" if embedding.device.startswith("cuda") else "float32",
            }
            identity_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            with HippoRAG(global_config=config, embedding_model=embedding, index_identity=identity_digest) as rag:
                if args.mode == "sample":
                    rag.index(docs)
                elif not rag.graph.vcount():
                    raise RuntimeError("No persisted graph. Run sample first with this configuration.")
                result = rag.rag_qa(
                    queries=questions, gold_docs=gold_docs,
                    gold_answers=[row["answer"] for row in samples],
                )
                if not rag.graph.vcount() or not rag.graph.ecount():
                    raise RuntimeError("The graph must contain nodes and edges.")
                report = {
                    "mode": args.mode, "dataset": args.dataset, "device": embedding.device,
                    "corpus_count": len(docs), "question_count": len(questions),
                    "graph_nodes": rag.graph.vcount(), "graph_edges": rag.graph.ecount(),
                    "index_identity": identity, "results": result,
                }
            report["upstream_commit"] = (ROOT / "upstream_revision.txt").read_text(encoding="utf-8").strip()
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report_path = output / f"{args.mode}_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        print(f"PASS: {args.mode}; report: {report_path}")
    finally:
        embedding.close()


if __name__ == "__main__":
    main()
