"""Replay three failed NER replies from SQLite without API calls or cache writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/HippoRAG/src"))

FAILED = {
    "chunk-3318f2e1a6da9f373729911fa4c89ef1",
    "chunk-bfb1662ff73c25de91ce58570323eeeb",
    "chunk-4cdf680a1d37e7a70e70dadddfa75439",
}


def cache_key(config, model, messages):
    from hipporag.llm.base import normalize_generation_token_params
    params = normalize_generation_token_params(model.llm_config.generate_params,
                                               {"max_new_tokens": config.openie_ner_max_tokens}, "_max_tokens")
    if config.response_format is not None:
        params["response_format"] = config.response_format
    params.update(messages=messages, _cache_schema=3,
                  _provider_class="hipporag.llm.openai_gpt.CacheOpenAI",
                  _endpoint=config.llm_base_url, _azure_api_version=config.azure_api_version,
                  _azure_chat_deployment=config.azure_chat_deployment)
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/server_native")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from settings import load_settings
    settings = load_settings()
    from hipporag.information_extraction.openie_openai import OpenIE
    from hipporag.llm.openai_gpt import CacheOpenAI
    from hipporag.utils.config_utils import BaseConfig
    from hipporag.prompts import PromptTemplateManager
    # Read the same settings and environment overrides as business.py. This does
    # not instantiate the OpenAI client or load the embedding model.
    config = BaseConfig(llm_name=settings["llm_model"], llm_base_url=settings["llm_base_url"])
    model = CacheOpenAI.__new__(CacheOpenAI)
    model.global_config = config
    model.request_model_name = config.llm_name
    model.llm_name = config.llm_name
    model.llm_base_url = config.llm_base_url
    model._init_llm_config()
    output = ROOT / args.output_dir
    database = output / "banking/index/llm_cache" / (config.llm_name.replace("/", "_") + "_cache.sqlite")
    corpus = json.loads((output / "banking/corpus.json").read_text(encoding="utf-8"))
    prompts = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
    report = {"database": str(database), "mode": "read_only_no_api_calls", "chunks": []}
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        report["cache_rows"] = connection.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        for chunk in corpus:
            chunk_id = "chunk-" + hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()
            if chunk_id not in FAILED:
                continue
            messages = prompts.render(name="ner", passage=chunk["text"])
            key = cache_key(config, model, messages)
            cached = connection.execute("SELECT message, metadata FROM cache WHERE key=?", (key,)).fetchone()
            item = {"chunk_id": chunk_id, "source_id": chunk["source_id"], "cache_key": key, "cache_hit": cached is not None}
            if cached is not None:
                message, metadata_text = cached
                metadata = json.loads(metadata_text)
                fake = SimpleNamespace(global_config=config, infer=lambda **kwargs: (message, dict(metadata), True))
                result = OpenIE(fake).ner(chunk_id, chunk["text"])
                item.update(finish_reason=metadata.get("finish_reason"),
                            parse_error=result.metadata.get("error"),
                            entity_count=len(result.unique_entities),
                            response=message)
            report["chunks"].append(item)
    report["located_failed_chunks"] = len(report["chunks"])
    path = output / "ner_cache_diagnosis.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in report["chunks"]:
        # Full replies remain in the report; print enough to identify JSON shape.
        preview = dict(item)
        if "response" in preview:
            preview["response"] = preview["response"][:1500]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(json.dumps({"cache_rows": report["cache_rows"],
                      "located_failed_chunks": report["located_failed_chunks"], "report": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
