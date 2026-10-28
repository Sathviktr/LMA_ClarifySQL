"""How far the team's hand work in annotation/ has got, and what is left (see annotation/README.md).

Usage:  python -I scripts/annotation_status.py
"""

from __future__ import annotations

from clarifysql.episodes.review import ANNOTATION, DECISIONS, cell, read_sheet


def report(task: str, name: str, done, invalid=lambda r: False, extra=lambda rows: "") -> bool:
    if not (ANNOTATION / name).exists():
        print(f"  ----  {task:<26} {name:<30} not created yet")
        return False
    rows = read_sheet(name)
    n_done, n_bad = sum(map(done, rows)), sum(map(invalid, rows))
    complete = n_done == len(rows) and not n_bad
    note = f", {n_bad} with an invalid value" if n_bad else ""
    print(f"  {'DONE' if complete else 'TODO'}  {task:<26} {name:<30} {n_done:>3}/{len(rows):<3} filled{note}"
          f"{extra(rows)}")
    return complete


def main() -> None:
    def decision(r):
        return cell(r, "decision").lower()

    def cleaned(r):
        return cell(r, "cleaned_decision").lower()

    def verdict(r):
        return cell(r, "verdict").lower()

    print(f"Hand work in {ANNOTATION}")
    for split in ("validation", "test"):
        print(f"{split}:")
        report("1 edit the requests", f"{split}_edit.csv", lambda r: bool(cell(r, "editor")))
        for who in ("A", "B"):
            report(f"2 blind labels ({who})", f"{split}_label_{who}.csv", lambda r: decision(r) in DECISIONS,
                   lambda r: bool(decision(r)) and decision(r) not in DECISIONS)
        if (ANNOTATION / f"{split}_label_A.csv").exists() and not (ANNOTATION / f"{split}_disagreements.csv").exists():
            name = f"{split}_disagreements.csv"
            print(f"  ----  {'3 resolve disagreements':<26} {name:<30} after both label sheets: "
                  f"python -I scripts/kappa.py {split}")
        else:
            report("3 resolve disagreements", f"{split}_disagreements.csv",
                   lambda r: cell(r, "resolved_decision").lower() in DECISIONS + ("drop",),
                   lambda r: bool(cell(r, "resolved_decision")) and
                   cell(r, "resolved_decision").lower() not in DECISIONS + ("drop",))
    print("other sheets:")
    report("4 clean the CoSQL slice", "cosql_dev_slice.csv",
           lambda r: cleaned(r) in DECISIONS + ("drop",) and bool(cell(r, "annotator")),
           lambda r: bool(cleaned(r)) and cleaned(r) not in DECISIONS + ("drop",),
           lambda rows: f"; {sum(cleaned(r) in DECISIONS for r in rows)} kept (40-60 needed)")
    report("5 audit the training data", "train_audit.csv",
           lambda r: verdict(r) in ("keep", "drop") and bool(cell(r, "auditor")),
           lambda r: bool(verdict(r)) and verdict(r) not in ("keep", "drop"),
           lambda rows: f"; {sum(verdict(r) == 'drop' for r in rows)} marked drop")
    print("A row counts as filled only when it is signed (editor / annotator / auditor) where the sheet asks for it.")


if __name__ == "__main__":
    main()
