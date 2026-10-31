"""Restore compressed experiment evidence without replacing differing local files."""
import gzip
import hashlib
from pathlib import Path


def restore(root):
    data = (root / "data").resolve()
    restored = []
    for source in sorted(data.rglob("*.json.gz")):
        target = source.with_suffix("").resolve()
        if not target.is_relative_to(data):
            raise ValueError("Evidence target escaped the data directory")
        raw = gzip.decompress(source.read_bytes())
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError(f"Refusing to replace differing local evidence: {target.name}")
        else:
            target.write_bytes(raw)
        restored.append({"path": target.relative_to(root).as_posix(),
                         "sha256": hashlib.sha256(raw).hexdigest()})
    return restored


if __name__ == "__main__":
    for row in restore(Path(__file__).resolve().parents[1]):
        print(row["path"], row["sha256"])
