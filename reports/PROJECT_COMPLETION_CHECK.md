# ClarifySQL project completion check

Checked on 8 November 2026 against the code, historical and final reports, recorded experiments,
annotation sheets, tests and learning materials.

The core implementation and experiments reported in the final report have supporting evidence.
Some broader demo plans are only partly demonstrated. Known failures and delivery limits remain explicit.

| Work | Evidence | Finding |
|---|---|---|
| Data preparation and review | `data/splits/`, `data/episodes/`, `annotation/`, `scripts/annotation_status.py` | Complete: 307 train, 53 validation and 188 test episodes; 602/101 policy states. Required dataset-review cells are filled. |
| Database separation | Episode and split files | Episode train, validation and test database sets are disjoint. The public final-test databases also had an earlier writer diagnostic. |
| Tools and read-only execution | `src/clarifysql/tools/`, `src/clarifysql/db/`, safety tests | Implemented and checked on software fixtures. Passing fixtures do not establish every possible model-driven safety outcome. |
| Agent and simulated user | `src/clarifysql/agent/`, `src/clarifysql/simuser/`, `src/clarifysql/demo/` | Inspection, clarification, memory, execution, abstention, budgets and repair paths exist. Semantic intent checking remains limited. |
| Training and selection | `results/train/`, `logs/`, notebooks 05–06 | Main seeds 1/2 and the no-slot adapter are recorded. Selection uses validation. Adapter weights remain in the saved external phase6 output. |
| Main final comparisons | `results/final/test_*/` | All eight real-model conditions contain 188 episodes each. Correctness, costs and recorded summary calculations match the report. |
| CoSQL evaluation | `results/final/cosql_*/` | Both real-model conditions contain the 59 reviewed turns; mixed generalization is reported. |
| Required ablations | A2 and A3 traces/manifests | Complete. A3 supplies full context while retaining inspection tools; it is not a strict tool-removal experiment. |
| Bootstrap analysis | `results/final/analysis/paired_bootstrap.json` | Reported database-paired intervals are supported. The small number of seeds and databases remains a limitation. |
| Question and simulator review | Seven CSVs in `annotation/evaluation/`; `human_audits.json`; `review_validation.json` | Returned sheets are complete and scored. Identical question ratings, coordinator-confirmed independence and rating-interpretation limits are disclosed. |
| Scripted model demo | `results/demo/checkpoint8/received.zip` and its analysis | Seven outcomes reproduce; original 5/7 is preserved. Both age-cutoff cases fail, and a placement answer is a false positive exposed by two variants. |
| Basic browser check | `results/demo/browser*_2026-11-08.json` | Three displayed paths are documented: department lookup, clarification exchange and missing-email explanation. |
| Historical source verification | `scripts/verify_final.py`, `scripts/analyze_demo.py` | All ten benchmark conditions and seven original demo cases verify locally. A full Git history is needed for the historical source checks. |
| Software checks | Audit record linked below | 210 CPU tests pass together; one training-loop test passes separately with one CPU thread and MKLDNN disabled. |
| Final report | `ClarifySQL_Final_Report.tex` and `.pdf` | Updated with measured results, reviews, prior benchmark use, reproducibility limits and known failures. Historical mid-report results remain historical. |
| Learning materials | Local `learning/phase1/`–`phase7/`, `learning/pdfs/`, `learning/final/` | Phase explanations and historical status notes are reviewed; the complete-project guide gives a current synthesis. Learning files remain local. |

The machine-readable check record is
[`repository_audit_2026-11-08.json`](../results/final/analysis/repository_audit_2026-11-08.json).
The checks reused the installed environment and local benchmark data. They did not repeat GPU inference,
retrain adapters, install a fresh environment or rerun the official database-variant queries.

## Plans that should not be called completed model demonstrations

The mid-report proposed a broader demo with a partial reply, SQL repair and blocked write. The repository
contains software fixtures for repair and write blocking, and repair events appear in benchmark traces.
It does not contain a successful real-model browser demonstration of every proposed path. The real
age-and-marks cases execute before resolving the age cutoff. These are recorded limitations, not hidden passes.

Extra ablations without memory, intent checks or repair were optional in the mid-report. No results are
claimed for those unrun conditions.

## Remaining delivery checks and reproduction limits

1. **External weights:** confirm that the instructor can access phase6 run `363373340`, containing
   `final_adapters/sft_s1`, `sft_s2`, `a2_s1` and `choice.json`. Metrics ZIPs do not include those weights.
2. **Base-model identity:** measured runs log a consistent revision, but the current loader and resume
   fingerprint do not enforce that revision for a future run. Exact reruns need the recorded base weights
   as well as the source, data and adapters. Automatic revision enforcement is a future improvement.
3. **Review reproduction:** the anonymous system key remains private on the coordinator's machine.
   Recomputing system-level human-review summaries requires that key; it is intentionally absent from a public clone.
4. **Final hand-in:** read the final PDF and submit the report, repository link and accessible weights link
   using the course's submission instructions.

The final episode comparison uses public Spider development databases. All 20 were used in a prior
1,034-question writer diagnostic, and possible overlap with base-model pretraining cannot be ruled out.
This does not change the disjoint policy-episode split, but it limits claims about completely unseen data.
