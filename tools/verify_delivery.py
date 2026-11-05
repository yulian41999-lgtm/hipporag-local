"""Read-only integrity check, runs with standard-library Python on the server."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "delivery_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = (ROOT / row["path"]).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            raise ValueError("Unsafe delivery path") from None
        if not path.is_file():
            raise ValueError("Missing or unsafe delivery path")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024**2), b""):
                digest.update(block)
        if path.stat().st_size != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError("Delivery mismatch: " + row["path"])
    print("PASS: " + str(len(manifest["files"])) + " delivered files match SHA-256 manifest")


if __name__ == "__main__":
    main()
