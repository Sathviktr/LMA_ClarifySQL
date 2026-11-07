# Checkpoint 7: measured findings

These are the frozen final test results, not validation scores.
All eight conditions cover 188 episodes on 20 databases: 148 answerable and 40 unanswerable.

| System | Correct / 148 | Exec. % | Test-suite % | Decision F1 | Ask on clear % | Correct abstention / 40 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 84 | 56.76 | 46.62 | 0.2207 | 0.00 | 0 |
| B2 | 89 | 60.14 | 47.30 | 0.2207 | 0.00 | 0 |
| B3 | 94 | 63.51 | 54.05 | 0.7850 | 26.58 | 35 |
| SFT 1 | 114 | 77.03 | 67.57 | 0.8790 | 2.53 | 29 |
| SFT 2 | 112 | 75.68 | 65.54 | 0.9257 | 0.00 | 29 |
| A2 | 115 | 77.70 | 68.24 | 0.8923 | 0.00 | 28 |
| A3 | 108 | 72.97 | 62.16 | 0.8533 | 0.00 | 28 |
| Oracle | 116 | 78.38 | 68.92 | 0.2207 | 0.00 | 0 |

## Primary comparison: training versus prompting

Intervals resample entire databases, use 5,000 paired draws and preserve each seed separately.
The two-seed average uses the same draws; it does not count the questions twice.

| Metric | Comparison | Difference | 95% paired interval |
|---|---|---:|---:|
| execution_accuracy | Seed 1 minus B3 | +13.5135 percentage points | [+5.4795, +20.6255] |
| execution_accuracy | Seed 2 minus B3 | +12.1622 percentage points | [+5.4688, +17.9015] |
| execution_accuracy | Two-seed mean minus B3 | +12.8378 percentage points | [+5.5545, +19.0757] |
| decision_macro_f1 | Seed 1 minus B3 | +0.0940 F1 | [+0.0350, +0.1623] |
| decision_macro_f1 | Seed 2 minus B3 | +0.1407 F1 | [+0.0880, +0.2050] |
| decision_macro_f1 | Two-seed mean minus B3 | +0.1173 F1 | [+0.0654, +0.1795] |

Both seeds improve execution and decision F1 on this test; the paired intervals above exclude zero.
This supports the primary comparison within the experiment and its database-resampling assumptions.
It does not establish general performance on unseen domains or validate the simulated user.

## Important tradeoffs and exploratory comparisons

- Correct abstention is lower for both SFT seeds than for B3. The system is better at answering but is less reliable at rejecting missing-information requests under the automatic metric.
- A2 solves one more answerable case than SFT seed 1. Its paired execution and decision intervals include zero. Explicit slots have not demonstrated an execution benefit.
- A3 does not improve the point estimates. Its execution interval touches zero; its decision interval includes zero. This is limited evidence about this full-context variant, not proof that retrieval is always necessary.
- A single test case is about 0.68 percentage points on the 148 answerable episodes.
- The oracle is a reference run, not a mathematical upper bound.

## Where the execution gain comes from

| Request type | Cases | B3 correct | SFT seed 1 correct | SFT seed 2 correct |
|---|---:|---:|---:|---:|
| clear | 79 | 56 | 67 | 68 |
| ambiguous | 47 | 31 | 35 | 34 |
| two_slot | 8 | 4 | 4 | 3 |
| definition | 14 | 3 | 8 | 7 |

The gains include fewer unnecessary questions on clear requests and better definition handling.
The partly answered cases remain difficult. Question usefulness still needs the blinded review.

## Real dialogue slice (CoSQL)

| System | Correct / 59 | Decision F1 | Clarify recall / 13 |
|---|---:|---:|---:|
| b3 | 26 | 0.3740 | 8 |
| sft_s1 | 40 | 0.4560 | 3 |
| Always execute (label-only baseline) | 45 | 0.2883 | 0 |

SFT improves overall decisions on the reviewed slice but detects fewer of its genuine clarification cases.
Its accuracy is still below the always-execute baseline because the slice is dominated by execute labels; its macro-F1 is higher. The accuracy gain over B3 alone does not establish strong natural-dialogue generalization.
The single abstention case cannot support a general abstention claim. This slice is exploratory.

## Cost and output reliability

| System | Total model calls | Total input tokens | Generation seconds |
|---|---:|---:|---:|
| b3 | 862 | 605,038 | 1198.2 |
| sft_s1 | 570 | 216,401 | 528.5 |
| sft_s2 | 552 | 207,348 | 558.2 |
| a2 | 573 | 205,609 | 479.5 |
| a3 | 525 | 273,484 | 664.9 |

These totals include policy, writer and simulated-user inference. Generation seconds are elapsed generation time, not CUDA kernel profiling.
The two-GPU inference cell took 40.6 minutes. CPU scoring time is not included in that cell measurement.
Invalid outputs and limits should be reported per system; one successful retry does not erase the original invalid attempt.

## Still required

Complete the independent question-usefulness and simulator reviews, run the live demo, and write the final report with the tradeoffs above.
Automatic missing-information and slot scores use lexical checks and have not been validated by those human reviews.
