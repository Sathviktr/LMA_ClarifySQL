# Final test results

All conditions cover the same reviewed test episodes.

| System | Execution accuracy | Decision macro-F1 | Test-suite accuracy |
|---|---:|---:|---:|
| b1 | 0.5676 | 0.2207 | 0.46621621621621623 |
| b2 | 0.6014 | 0.2207 | 0.47297297297297297 |
| oracle | 0.7838 | 0.2207 | 0.6891891891891891 |
| b3 | 0.6351 | 0.785 | 0.5405405405405406 |
| sft_s1 | 0.7703 | 0.879 | 0.6756756756756757 |
| sft_s2 | 0.7568 | 0.9257 | 0.6554054054054054 |
| a2 | 0.777 | 0.8923 | 0.6824324324324325 |
| a3 | 0.7297 | 0.8533 | 0.6216216216216216 |

Confirmatory comparisons: SFT versus B3 on execution accuracy and decision macro-F1.
Database-level intervals and both seeds are in paired_bootstrap.json.
Question usefulness and simulator accuracy require the separate human review sheets.
