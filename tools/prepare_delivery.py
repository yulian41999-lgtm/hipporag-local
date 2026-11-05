"""Create a clean server source archive; .dockerignore does not filter uploads."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "Dockerfile", ".dockerignore", ".env.example", ".gitignore", "README.md",
    "requirements.txt", "requirements-local.txt", "config.json", "business_config.json",
    "settings.py", "local_embedding.py", "business.py", "business_qa.py", "run.py",
    "probe_llm.py", "container_cli.py", "upstream_revision.txt",
    "compose.yaml", "compose.gpu.yaml", "compose.server.yaml", "compose.server-build.yaml",
}
DIRECTORIES = {"data", "models", "evaluation", "vendor", "docs", "tests", "tools"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "outputs", "cache", "dist"}


def excluded(relative):
    return any(p in EXCLUDED_PARTS or p.endswith(".egg-info") for p in relative.parts) or relative.suffix in {".pyc", ".log"} or (relative.name.startswith(".env") and relative.name != ".env.example")


def delivery_files():
    paths = [ROOT / f for f in FILES]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("A required top-level delivery file is missing")
    for directory in DIRECTORIES:
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and not excluded(p.relative_to(ROOT)))
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Delivery may only include regular files inside the project")
    return sorted(paths)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="Create a .tar file; without this argument only inspect the delivery set")
    args = parser.parse_args()
    paths = delivery_files()
    verification = json.loads((ROOT / "models/bge-m3.copy-verification.json").read_text(encoding="utf-8-sig"))
    for row in verification["files"]:
        if sha256(ROOT / "models/bge-m3" / row["path"]) != row["sha256"]:
            raise ValueError("Model hash mismatch: " + row["path"])
    gold_dir = ROOT / "data/banking/银行监管测试集"
    gold = gold_dir / "banking_qa_15.json"
    audit = json.loads((gold_dir / "content_audit.json").read_text(encoding="utf-8"))
    if sha256(gold) != audit["dataset_sha256"] or len(json.loads(gold.read_text(encoding="utf-8"))) != 15:
        raise ValueError("Gold dataset integrity check failed")
    total = sum(p.stat().st_size for p in paths)
    print(json.dumps({"files": len(paths), "size_gib": round(total / 1024**3, 3), "model_hashes_verified": True, "gold_question_count": 15, "secrets_included": False, "archive_created": bool(args.output)}))
    if args.output is None:
        return
    if args.output.suffix != ".tar":
        parser.error("Use a .tar output (model weights already dominate the archive size)")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "server_build_source_delivery_not_a_docker_image",
        "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths],
    }
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            archive.add(path, arcname="hipporag_local/" + path.relative_to(ROOT).as_posix(), recursive=False)
        data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        info = tarfile.TarInfo("hipporag_local/delivery_manifest.json")
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    checksum = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(checksum + "  " + output.name + "\n", encoding="ascii")
    print("Source archive: " + str(output))


if __name__ == "__main__":
    main()
