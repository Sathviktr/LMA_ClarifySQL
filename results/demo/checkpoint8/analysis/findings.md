# Checkpoint 8: measured findings

The original saved model run passes **5/7** fixed checks; that result is preserved.
The seven cases are a scripted demonstration, not a benchmark or human study. All seven result records reproduce on CPU.

| Case | Original result | Observation |
|---|---|---|
| demo-clear | PASS | Correct department names. |
| demo-definition | PASS | Retrieves the above-80 glossary definition and returns the correct names. |
| demo-ambiguous | PASS | Returns Mathematics after the fixed average-mark reply, although its offered age/ID options are poorly grounded. |
| demo-ranking-placements | PASS | Returns Computing on the seed data, but its SQL fails the additional placement variants below. |
| demo-two-slot | FAIL | Does not ask; uses marks above 80 and guesses age above 18 instead of obtaining the missing age cutoff. |
| demo-two-slot-age20 | FAIL | Does not ask, so the changed age-20 reply is never supplied. Repeats the same incorrect query. |
| demo-unanswerable | PASS | Correctly abstains and identifies missing email addresses. |

## Placement SQL: a matching answer can still use the wrong rule

The prediction orders by COUNT(p.student_id)/COUNT(s.id). Every student has a placement row, including those with placed=0, so all departments receive rate 1. Computing wins the tie on this seed database.
The intended rate uses the placed indicator, for example SUM(p.placed)/COUNT(*), with floating-point division.

The additional checks change only the fictional placed flags and retain the original predicted SQL:

| Variant | Expected | Predicted | Match |
|---|---|---|---|
| mathematics_best | Mathematics | Computing | False |
| physics_best | Physics | Computing | False |

These are post-hoc counterexamples, not a new benchmark score. The original 5/7 remains unchanged.

## Other limitations

- Both ranking questions offer age or ID; departments has no age column. A cooperative scripted reply supplies a better criterion than the offered choices. Passing execution does not establish useful question wording.
- The glossary retrieval matches strong marks to high marks, but older still has no supplied cutoff. The policy executes prematurely and the writer supplies 18. Even allowing the glossary match does not resolve age.
- The missing numeric value cannot be caught by a check that only verifies numbers already present in replies, because no reply was requested.
- Gradio construction succeeded. Browser clicks, real replies and sharing have not been demonstrated by this archive.

The run uses 21 model calls and 56.4 elapsed generation seconds; the model load and installation are separate.
Source and adapter hashes match the recorded code and checkpoint-7 seed 1. The base revision matches the frozen test run.

## Remaining work

Try the interactive interface, retain these model limitations in the final report, and keep the independent human study status separate. No model retraining or checkpoint-7 rerun is needed to reproduce this analysis.
