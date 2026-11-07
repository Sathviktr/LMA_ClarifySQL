"""Run the seven scripted demo checks with the selected trained adapter (GPU)."""

import argparse
from pathlib import Path

from clarifysql.demo.checks import run_demo_checks
from clarifysql.demo.university import create_database


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/checkpoint8_demo"))
    a = ap.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("A GPU is required for the trained model. Run notebook 08 on Kaggle.")
    from clarifysql.llm.backend import HFBackend
    model = HFBackend(adapter_path=str(a.adapter))
    database = create_database(a.output / "university.sqlite")
    summary = run_demo_checks(model, database, a.output / "checks", adapter_path=a.adapter)
    print(f"Saved {summary['cases']} cases ({summary['passed']} passed) in {a.output / 'checks'}")


if __name__ == "__main__":
    main()
