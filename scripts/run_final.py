"""Checkpoint 7: resumable final episodes or the reviewed CoSQL first-decision slice."""

import argparse
from pathlib import Path
from clarifysql.eval.final_run import run_condition, SYSTEMS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=SYSTEMS, required=True)
    ap.add_argument("--split", choices=["test", "validation", "cosql"], default="test")
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--output", type=Path, default=Path("results/final"))
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--testsuite", action="store_true")
    a = ap.parse_args()
    print(run_condition(a.system, a.split, a.output, adapter=a.adapter, name=a.name,
                        backend_name=a.backend, batch_size=a.batch_size, chunk_size=a.chunk_size,
                        limit=a.limit, use_testsuite=a.testsuite))


if __name__ == "__main__":
    main()
