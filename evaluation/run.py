"""Evaluate saved results; never run retrieval with access to gold answers."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import ROOT, load_settings
from evaluation.tokenization import TOKENIZER_VERSION, rouge_l
from evaluation.judge import JSONJudge
from evaluation.metrics import context_relevancy, evidence_recall, answer_correctness


def validate_input(rows, allow_partial=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation input must be a nonempty array")
    ids = []
    for row in rows:
        for field in ("id", "question", "question_type", "ground_truth", "generated_answer"):
            if not isinstance(row.get(field), str):
                raise ValueError("Missing string field: " + field)
        for field in ("context", "evidence"):
            if not isinstance(row.get(field), list) or any(not isinstance(x, str) for x in row[field]):
                raise ValueError(field + " must be a list of strings")
        if not row["ground_truth"].strip() or not row["evidence"]:
            raise ValueError("Reference answer and evidence must be nonempty")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs")
    if not allow_partial:
        folder = ROOT / "data/banking/银行监管测试集"
        gold_path = folder / "banking_qa_15.json"
        gold = {row["id"]: row for row in json.loads(gold_path.read_text(encoding="utf-8"))}
        audit = json.loads((folder / "content_audit.json").read_text(encoding="utf-8"))
        if hashlib.sha256(gold_path.read_bytes()).hexdigest() != audit["dataset_sha256"]:
            raise ValueError("Gold dataset differs from audited version")
        if set(ids) != set(gold) or len(rows) != 15:
            raise ValueError("Formal evaluation must include all 15 gold IDs")
        for row in rows:
            reference = gold[row["id"]]
            expected_evidence = [s["quote"] for s in reference["evidence_sources"]]
            if (row["question"], row["ground_truth"], row["evidence"], row["question_type"]) != (
                reference["question"], reference["answer"], expected_evidence, reference["question_type"]
            ):
                raise ValueError("Evaluation reference differs from frozen gold: " + row["id"])


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups["overall"].append(row)
        groups[row["question_type"]].append(row)
    output = {}
    for name, rows in groups.items():
        metrics = {}
        for metric in rows[0]["metrics"]:
            valid = [r["metrics"][metric]["score"] for r in rows if r["metrics"][metric]["status"] == "ok"]
            metrics[metric] = {
                "mean": sum(valid) / len(valid) if valid else None,
                "expected_n": len(rows), "valid_n": len(valid), "failed_n": len(rows) - len(valid),
            }
        output[name] = {"metrics": metrics, "empty_answer_n": sum(not r["generated_answer"].strip() for r in rows)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Label an isolated smoke test; never a formal 15-question run")
    args = parser.parse_args()
    settings = load_settings()
    config = json.loads((ROOT / "evaluation/config.json").read_text(encoding="utf-8"))
    if config["rouge_tokenizer"] != TOKENIZER_VERSION or config["correctness_weights"] != [0.75, 0.25]:
        raise ValueError("Metric configuration must match the implemented, versioned formulas")
    for name, env in {"judge_model": "HIPPO_JUDGE_MODEL", "judge_base_url": "HIPPO_JUDGE_BASE_URL", "judge_api_key_env": "HIPPO_JUDGE_API_KEY_ENV"}.items():
        config[name] = os.environ.get(env, config[name])
    raw = args.input.read_bytes()
    rows = json.loads(raw.decode("utf-8-sig"))
    validate_input(rows, args.allow_partial)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or ROOT / "outputs/evaluation" / stamp
    output.mkdir(parents=True, exist_ok=True)
    if (output / "scores.json").exists():
        raise FileExistsError("Choose a new output directory to preserve previous results")
    from hipporag.utils.config_utils import BaseConfig
    from local_embedding import LocalBGEEmbedding
    encoder_settings = dict(settings, max_seq_length=config["embedding_max_seq_length"], batch_size=1)
    encoder_config = BaseConfig(embedding_model_name=settings["embedding_model_id"], embedding_return_as_normalized=True)
    embedding = LocalBGEEmbedding(encoder_config, encoder_settings)
    judge = JSONJudge(config, output / "judge_calls.jsonl")
    results = []
    try:
        for row in rows:
            record = {"id": row["id"], "question_type": row["question_type"], "generated_answer": row["generated_answer"], "metrics": {}}
            calls = {
                "rouge_l": lambda: (rouge_l(row["generated_answer"], row["ground_truth"]), {"tokenizer": TOKENIZER_VERSION}),
                "context_relevancy": lambda: context_relevancy(row["question"], row["context"], judge, row["id"]),
                "evidence_recall": lambda: evidence_recall(row["question"], row["context"], row["evidence"], judge, row["id"]),
                "answer_correctness": lambda: answer_correctness(row["question"], row["generated_answer"], row["ground_truth"], judge, embedding, row["id"]),
            }
            for name in config["metrics"]:
                try:
                    score, detail = calls[name]()
                    record["metrics"][name] = {"score": float(score), "status": "ok", "details": detail}
                except Exception as exc:
                    record["metrics"][name] = {"score": None, "status": "error", "error": type(exc).__name__}
                    if isinstance(exc, ValueError):
                        record["metrics"][name]["reason"] = str(exc)
            results.append(record)
            (output / "scores.partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            print(f"Evaluated {len(results)}/{len(rows)}: {row['id']}", flush=True)
    finally:
        judge.close()
        embedding.close()
    report = {
        "run_scope": "smoke_or_partial" if args.allow_partial else "formal_15",
        "profile": config, "input_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation": "GraphRAG-Bench formulas; Chinese prompts, explicit tokenizer and strict complete-ID validation",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ["evaluation/metrics.py", "evaluation/judge.py", "evaluation/tokenization.py", "business_qa.py"]},
        "scoring_embedding": {"model": settings["embedding_model_id"], "revision": settings["embedding_revision"], "device": embedding.device, "normalized": True},
        "retrieval_embedding_max_seq_length": settings["max_seq_length"],
        "scoring_embedding_max_seq_length": config["embedding_max_seq_length"],
        "generation_model": settings["llm_model"],
        "same_model_judge": config["judge_model"] == settings["llm_model"],
        "aggregation": "per-question macro mean over valid scores; counts mandatory",
        "scores": results, "summary": summarize(results),
    }
    (output / "scores.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved: " + str(output / "scores.json"))
    if any(m["status"] != "ok" for r in results for m in r["metrics"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
