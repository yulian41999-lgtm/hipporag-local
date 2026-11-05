"""Load local configuration without printing credentials."""
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


def load_settings(config_path=None):
    vendor_source = ROOT / "vendor" / "HippoRAG" / "src"
    if vendor_source.is_dir() and str(vendor_source) not in sys.path:
        sys.path.insert(0, str(vendor_source))
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip())
    path = Path(config_path or os.environ.get("HIPPO_CONFIG_PATH", "config.json"))
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    overrides = {
        "embedding_path": "HIPPO_EMBEDDING_PATH",
        "llm_base_url": "HIPPO_LLM_BASE_URL",
        "llm_model": "HIPPO_LLM_MODEL",
        "output_dir": "HIPPO_OUTPUT_DIR",
        "device": "HIPPO_DEVICE",
    }
    for key, variable in overrides.items():
        if os.environ.get(variable):
            config[key] = os.environ[variable]
    for key, variable in {"batch_size": "HIPPO_BATCH_SIZE", "max_seq_length": "HIPPO_MAX_SEQ_LENGTH", "entity_batch_size": "HIPPO_ENTITY_BATCH_SIZE", "fact_batch_size": "HIPPO_FACT_BATCH_SIZE"}.items():
        if os.environ.get(variable):
            config[key] = int(os.environ[variable])
    model_path = Path(config["embedding_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    config["embedding_path"] = str(model_path.resolve())
    host = urlparse(config["llm_base_url"]).hostname
    bypass = [os.environ.get("NO_PROXY", ""), "localhost", "127.0.0.1", host or ""]
    os.environ["NO_PROXY"] = ",".join(item for item in bypass if item)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "cache" / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "cache" / "tiktoken"))
    return config
