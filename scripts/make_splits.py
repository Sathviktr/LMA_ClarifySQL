"""Create data/splits/splits.json (database-level train / validation / test split).

Usage:  python -I scripts/make_splits.py [--seed 2026]
Running it again with the same seed and data reproduces the file exactly.
"""

from __future__ import annotations

import argparse

from clarifysql.data.splits import SEED, make_splits, write_splits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    splits = make_splits(args.seed)
    path = write_splits(splits)
    print(f"eligible validation pool: {splits['rules']['eligible_count']} databases")
    print("validation databases:", ", ".join(splits["validation"]))
    for name, st in splits["stats"].items():
        print(f"{name:<10} " + "  ".join(f"{k}={v}" for k, v in st.items()))
    print("TrustSQL test databases:", len(splits["test_extras"]["trustsql_infeasible_databases"]),
          "| CoSQL dev databases inside test:", splits["test_extras"]["cosql_dev_databases_all_in_test"])
    print(f"written to {path}")


if __name__ == "__main__":
    main()
