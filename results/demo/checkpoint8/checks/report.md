# Demo checks

Backend: hf. Passed: 5/7.
Predefined replies are used; the interactive interface has not been tested by this script.

| Case | Result | Failed checks |
|---|---|---|
| demo-clear | PASS | - |
| demo-definition | PASS | - |
| demo-ambiguous | PASS | - |
| demo-ranking-placements | PASS | - |
| demo-two-slot | FAIL | expected_first_decision, result_matches_reference, reply_recorded |
| demo-two-slot-age20 | FAIL | expected_first_decision, result_matches_reference, reply_recorded |
| demo-unanswerable | PASS | - |
