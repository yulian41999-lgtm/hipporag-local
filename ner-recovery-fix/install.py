"""Install the checked NER fix and snapshot LLM cache into a fresh output root."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-output", default="outputs/server_native")
    parser.add_argument("--target-output", default="outputs/server_native_full")
    args = parser.parse_args()
    root = args.project_root.resolve()
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    target_file = inside(root, manifest["path"])
    before = target_file.read_bytes()
    if digest(before) != manifest["original_sha256"]:
        raise ValueError("Server OpenIE file differs from the diagnosed version; refusing replacement")
    replacement = (bundle / "openie_openai.py").read_bytes()
    if digest(replacement) != manifest["patched_sha256"]:
        raise ValueError("Patch checksum failed")
    backup = target_file.with_name(target_file.name + ".before-ner-recovery.bak")
    if backup.exists():
        raise FileExistsError("Backup already exists; inspect prior installation")
    source_root = inside(root, args.source_output)
    target_root = inside(root, args.target_output)
    if target_root.exists():
        raise FileExistsError("Use a fresh target output directory")
    databases = list((source_root / "banking/index/llm_cache").glob("*.sqlite"))
    if len(databases) != 1:
        raise ValueError("Expected one LLM cache database")
    source_db = databases[0]
    with sqlite3.connect(source_db.resolve().as_uri() + "?mode=ro", uri=True) as source:
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Source cache integrity check failed")
        source_count = source.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        target_db = target_root / "banking/index/llm_cache" / source_db.name
        target_db.parent.mkdir(parents=True)
        with sqlite3.connect(target_db) as destination:
            source.backup(destination)
            if destination.execute("SELECT COUNT(*) FROM cache").fetchone()[0] != source_count:
                raise ValueError("Cache snapshot count mismatch")
    backup.write_bytes(before)
    target_file.write_bytes(replacement)
    report = {"status": "PASS", "cache_rows_preserved": source_count,
              "source_output": args.source_output, "target_output": args.target_output,
              "original_database_modified": False, "openie_backup": str(backup)}
    (target_root / "recovery_setup.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
