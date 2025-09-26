from pathlib import Path
import json

def ensure_dirs(base: Path):
    (base / "parsed" / "images").mkdir(parents=True, exist_ok=True)
    (base / "parsed" / "page_images").mkdir(parents=True, exist_ok=True)
    (base / "parsed").mkdir(parents=True, exist_ok=True)
    (base / "index").mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows
