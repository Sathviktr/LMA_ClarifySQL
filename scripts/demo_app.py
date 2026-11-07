"""Launch the fictional university demo with the selected real adapter (GPU)."""

import argparse
from clarifysql.config import outputs_dir
from clarifysql.demo.university import create_database
from clarifysql.demo.app import create_app, launch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    from clarifysql.llm.backend import HFBackend
    db = create_database(outputs_dir() / "demo/university.sqlite")
    launch(create_app(HFBackend(adapter_path=a.adapter), db), share=a.share)


if __name__ == "__main__":
    main()
