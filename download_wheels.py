"""Download pinned runtime wheels with certifi and verify PyPI SHA-256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import httpx
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

PACKAGES = {
    "torch": "2.5.1", "transformers": "4.45.2", "litellm": "1.73.1",
    "sentence-transformers": "3.3.1", "python-igraph": "0.11.8",
    "igraph": "0.11.8", "tiktoken": "0.7.0", "tenacity": "8.5.0",
    "boto3": "1.43.121", "botocore": "1.43.121", "jmespath": "1.1.0",
    "s3transfer": "0.19.3", "nest-asyncio": "1.6.0",
    "huggingface-hub": "0.36.2", "tokenizers": "0.20.3",
    "importlib-metadata": "9.0.1", "jsonschema": "4.26.0",
    "jsonschema-specifications": "2025.9.1", "referencing": "0.37.0",
    "rpds-py": "0.30.0", "zipp": "4.1.2", "sympy": "1.13.1",
    "mpmath": "1.3.0", "python-dateutil": "2.9.0.post0",
    "texttable": "1.7.2", "hf-xet": "1.2.0",
    "datasets": "4.4.1", "dill": "0.4.0", "multiprocess": "0.70.18",
    "fsspec": "2025.10.0",
}
ROOT = Path(__file__).resolve().parent / "cache" / "wheels"
TAGS = {tag: index for index, tag in enumerate(sys_tags())}


def download(item):
    name, version = item
    with httpx.Client(trust_env=False, timeout=60, follow_redirects=True) as client:
        response = client.get(f"https://pypi.org/pypi/{name}/{version}/json")
        response.raise_for_status()
        payload = response.json()
        candidates = []
        for entry in payload["urls"]:
            if entry["filename"].endswith(".whl"):
                tags = parse_wheel_filename(entry["filename"])[3]
                rank = min((TAGS[tag] for tag in tags if tag in TAGS), default=None)
                if rank is not None:
                    candidates.append((rank, entry))
        if not candidates:
            raise RuntimeError(f"No Windows Python 3.10 wheel: {name}=={version}")
        entry = min(candidates, key=lambda row: row[0])[1]
        target = ROOT / entry["filename"]
        digest = entry["digests"]["sha256"]
        if target.exists():
            with target.open("rb") as handle:
                cached_digest = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()
            if cached_digest == digest:
                return f"CACHED {name}=={version}"
        partial = target.with_suffix(".part")
        hasher = hashlib.sha256()
        if entry["size"] > 64 * 1024**2:
            with partial.open("wb") as handle:
                for start in range(0, entry["size"], 4 * 1024**2):
                    end = min(start + 4 * 1024**2, entry["size"]) - 1
                    response = client.get(entry["url"], headers={"Range": f"bytes={start}-{end}"})
                    response.raise_for_status()
                    if response.status_code != 206 or len(response.content) != end - start + 1:
                        raise RuntimeError("The wheel host did not honor the requested byte range.")
                    hasher.update(response.content)
                    handle.write(response.content)
                    print(f"{name}: {end + 1}/{entry['size']} bytes", flush=True)
        else:
            with client.stream("GET", entry["url"]) as stream, partial.open("wb") as handle:
                stream.raise_for_status()
                for chunk in stream.iter_bytes(1024 * 1024):
                    hasher.update(chunk)
                    handle.write(chunk)
        if hasher.hexdigest() != digest:
            raise RuntimeError(f"SHA-256 mismatch: {name}")
        partial.replace(target)
        return f"OK {name}=={version} ({target.stat().st_size // 1024**2} MiB)"


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(download, PACKAGES.items()):
            print(result, flush=True)
