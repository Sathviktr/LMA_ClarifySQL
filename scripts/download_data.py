"""Download Spider 1.0, CoSQL and TrustSQL (with their SQLite databases) into the data folder.

Usage (from the repository root):
    python -I scripts/download_data.py                 # all three datasets
    python -I scripts/download_data.py --only spider   # just one
    python -I scripts/download_data.py --only testsuite   # optional: test-suite evaluator databases

The data folder is clarifysql.config.data_dir() (set CLARIFYSQL_DATA to choose it). Each archive is saved
in <data>/downloads/ and extracted into its own folder <data>/<name>/. Its size and SHA-256 checksum are
recorded in <data>/manifest.json and copied to data/manifest.json in the repository, so every team member
can check they have byte-identical data.

If Google Drive refuses the automatic download ("too many users"), download the file in a browser from
the printed link, put it in <data>/downloads/ under the printed name, and run the script again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

from clarifysql.config import REPO_ROOT, data_dir

SOURCES = {
    "spider": {
        "gdrive_id": "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J",
        "archive": "spider_data.zip",
        "license": "CC BY-SA 4.0",
        "homepage": "https://yale-lily.github.io/spider",
    },
    "cosql": {
        "gdrive_id": "1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF",
        "archive": "cosql_dataset.zip",
        "license": "CC BY-SA 4.0",
        "homepage": "https://yale-lily.github.io/cosql",
    },
    "trustsql": {
        "gdrive_id": "19IpLSc2QncO2273E8z-lvU2ME9wEHIan",
        "archive": "trustsql_dataset.zip",
        "license": "CC BY 4.0",
        "homepage": "https://github.com/glee4810/TrustSQL",
    },
    # Optional (large): database variants for the test-suite evaluator, used for final test scoring.
    "testsuite": {
        "gdrive_id": "1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w",
        "archive": "testsuite_databases.zip",
        "license": "Apache-2.0 (evaluator); databases derived from Spider",
        "homepage": "https://github.com/taoyds/test-suite-sql-eval",
    },
}
DEFAULT = ("cosql", "spider", "trustsql")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path) -> int:
    """Extract a zip file, refusing entries that would land outside `dest` (a "zip slip")."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = member.filename
            parts = Path(name).parts
            if not parts or parts[0] == "__MACOSX" or Path(name).name.startswith("._"):
                continue
            target = (dest / name).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"unsafe path in {archive.name}: {name!r}")
            zf.extract(member, dest)
            count += 1
    return count


def download(name: str, root: Path) -> dict:
    src = SOURCES[name]
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / src["archive"]
    link = f"https://drive.google.com/file/d/{src['gdrive_id']}/view"
    if not archive.exists():
        import gdown

        print(f"[{name}] downloading {src['archive']} from Google Drive ...")
        tmp = archive.with_suffix(".part")
        result = gdown.download(id=src["gdrive_id"], output=str(tmp), quiet=False, resume=True)
        if not result or not tmp.exists():
            sys.exit(f"[{name}] automatic download failed. Download it in a browser:\n  {link}\n"
                     f"and save it as {archive}, then run this script again.")
        tmp.replace(archive)
    else:
        print(f"[{name}] using existing {archive}")
    out_dir = root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    n = safe_extract(archive, out_dir)
    print(f"[{name}] extracted {n} entries into {out_dir}")
    return {
        "archive": src["archive"],
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "license": src["license"],
        "source": link,
        "homepage": src["homepage"],
        "downloaded": dt.date.today().isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(SOURCES), default=list(DEFAULT),
                    help="datasets to fetch (default: cosql spider trustsql; 'testsuite' is optional)")
    args = ap.parse_args()
    root = data_dir()
    print(f"data folder: {root}")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    repo_manifest_path = REPO_ROOT / "data" / "manifest.json"
    expected = json.loads(repo_manifest_path.read_text(encoding="utf-8")) if repo_manifest_path.exists() else {}
    for name in args.only:
        manifest[name] = download(name, root)
        want = expected.get(name, {}).get("sha256")
        if want:
            same = want == manifest[name]["sha256"]
            print(f"[{name}] SHA-256 {'matches' if same else 'DIFFERS FROM'} the repository manifest")
    text = json.dumps(dict(sorted(manifest.items())), indent=2) + "\n"
    manifest_path.write_text(text, encoding="utf-8")
    (REPO_ROOT / "data").mkdir(exist_ok=True)
    (REPO_ROOT / "data" / "manifest.json").write_text(text, encoding="utf-8")
    print(f"manifest written to {manifest_path} and data/manifest.json")


if __name__ == "__main__":
    main()
