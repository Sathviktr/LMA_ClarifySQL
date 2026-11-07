"""Package the current source and small required data for Kaggle without publishing or model weights."""

import hashlib
import json
from pathlib import Path
import zipfile
from clarifysql.config import REPO_ROOT


def main():
    root = REPO_ROOT
    paths = []
    for folder in ("src", "scripts", "tests", "notebooks", "data/episodes", "data/splits"):
        paths += [p for p in (root / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix not in ('.pyc', '.sqlite', '.db')]
    paths += [root / p for p in ("pyproject.toml", "requirements.txt", "requirements-gpu.txt",
                                "requirements-demo.txt", "requirements-eval.txt",
                                "annotation/cosql_dev_slice.csv", "results/train/selection.json")]
    contents = {str(p.relative_to(root)).replace('\\', '/'): p.read_bytes() for p in sorted(set(paths))}
    manifest = {"purpose": "checkpoint7-source", "files": {name: hashlib.sha256(data).hexdigest()
                                                             for name, data in contents.items()}}
    out = root / "outputs/clarifysql_checkpoint7_source.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in contents.items():
            z.writestr(name, data)
        z.writestr("CHECKPOINT7_SOURCE.json", json.dumps(manifest, indent=2))
    print(out, f"({out.stat().st_size:,} bytes, {len(contents)} source/data files, no weights)")


if __name__ == "__main__":
    main()
