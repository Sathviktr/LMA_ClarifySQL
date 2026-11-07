# ClarifySQL: teammate review tasks and Git workflow

**Review status, 8 November 2026:** all seven sheets are complete and scored. Question reviewers are
Subhash (A) and Manish (B). See the [scores](../../results/final/analysis/human_audits.json) and
[validation record](../../results/final/analysis/review_validation.json). The instructions below document
the review procedure; do not regenerate or replace the completed sheets.

The training, final model experiments, ablations and three basic browser checks are complete.
The review covers the saved clarification questions and simulator replies.
The seven CSVs in this folder are the review sheets. No Kaggle run, GPU or Python setup is needed to rate them.

## 1. Choose your assigned role

Sathvik coordinates collection. Assign A and B to two different teammates and C to the third teammate.
Each person edits only their assigned files.

| Role | File(s) in this folder | Exact task | Branch to create |
|---|---|---|---|
| Question reviewer A | [question_usefulness_A.csv](question_usefulness_A.csv) | Independently rate all 120 question items | `review/question-a` |
| Question reviewer B | [question_usefulness_B.csv](question_usefulness_B.csv) | Independently rate the same 120 question items | `review/question-b` |
| Simulator reviewer C | The five `simulator_*.csv` files listed below | Check all 247 simulator decisions | `review/simulator-c` |

A and B must not split the 120 items: every item needs a separate rating from each person.

## 2. Get the files and start your branch

For a first checkout:

```powershell
git clone https://github.com/Sathviktr/LMA_ClarifySQL.git
cd LMA_ClarifySQL
```

If you already have the repository, open a terminal in its `LMA_ClarifySQL` folder instead.
Before starting, make sure your working tree has no unfinished edits, then update `main`:

```powershell
git switch main
git pull --ff-only origin main
```

Run **only the branch command for your assigned role**:

| Role | Command |
|---|---|
| A | `git switch -c review/question-a` |
| B | `git switch -c review/question-b` |
| C | `git switch -c review/simulator-c` |

If you have already created your branch, use `git switch` with its name without `-c` to continue.
If you cannot push to this repository, tell Sathvik your GitHub username so repository access can be arranged.

## 3. Rules while editing

1. Open your assigned CSV in Excel or LibreOffice Calc with UTF-8 encoding and comma-separated columns.
   Turn on text wrapping and widen the text columns to read complete cells.
2. Fill only the judgment columns explained below, `annotator` and `notes`.
3. Keep all existing rows, row order, IDs, column names, requests, schema and saved dialogue unchanged.
4. Put your own initials in `annotator` on every row you review.
5. Save with the original filename as **CSV UTF-8**, not an Excel workbook.
6. Finish your assigned sheet(s) before pushing your completed review. Use the commands in section 6.

Work independently. Until both question reviewers finish, A and B must not read each other's sheets,
review branches or pull-request diffs, simulator sheets, system-labelled traces or previous draft judgments.
Separate Git branches do not hide those files; independence depends on following this rule. The system key,
`question_key.csv`, stays on the coordinator's machine and is excluded from Git.

If an intended reference is unclear, write `reference unclear` in `notes` and ask Sathvik before finalizing
that row. Any reference clarification should go to both question reviewers without sharing their ratings.

## 4. Question reviewers A and B

Complete all **120 rows** in your assigned `question_usefulness_A.csv` or `question_usefulness_B.csv`.
For each row, read `request`, `intended_request`, `tables` and the entire `dialogue`.
Judge the clarification questions against the intended meaning, then fill these four columns with `1` or `0`:

| Column | Enter `1` when... | Enter `0` when... |
|---|---|---|
| `targets_uncertainty` | The question asks about a detail actually missing from the user's request. | It asks about an unrelated or already clear detail. |
| `schema_supported` | Its options and assumptions fit the supplied tables and columns. | It relies on unavailable information or unsuitable schema assumptions. |
| `not_repeated` | It avoids asking again for information already supplied. | It repeats an answered question or supplied detail unnecessarily. |
| `resolves_intent` | A cooperative user could provide the needed meaning by answering the question or sequence. | The questions leave the needed meaning unresolved or prevent a useful answer. |

Judge the criteria separately. A question can target the right topic but offer unsuitable options.
When several questions were asked, judge the sequence as a whole. A poor simulator reply does not
automatically make a well-phrased question bad.

If `dialogue` says `No clarification question was asked.`, enter **`0` in all four columns** and write
`No clarification question was asked.` in `notes`. Add your initials in `annotator` on every row.
Explain uncertain judgments in `notes`; do not reconcile your answers with the other reviewer.

Before committing, check that all 120 rows have four binary judgments and your initials.

## 5. Simulator reviewer C

Complete these five files:

| File | Rows to review |
|---|---:|
| [simulator_b3.csv](simulator_b3.csv) | 50 |
| [simulator_sft_s1.csv](simulator_sft_s1.csv) | 50 |
| [simulator_sft_s2.csv](simulator_sft_s2.csv) | 50 |
| [simulator_a2.csv](simulator_a2.csv) | 50 |
| [simulator_a3.csv](simulator_a3.csv) | 47 |
| **Total** | **247** |

A3 has only 47 available decisions. Do not add or duplicate rows to reach 50. Repeated episode IDs can
represent different simulator decisions; keep every existing row.

For each row:

1. Read `request`, `question` and `hidden_details`. Decide which hidden details the question asks about.
2. Enter their zero-based indices in `human_slot_indices`: the first detail is `0`, the second is `1`.
   Enter `0,1` in one cell if both are requested, or `none` if neither is requested.
3. Read `simulator_choice` and `simulator_reply`. Enter `1` in `correct` if the simulator selected and
   revealed all appropriate details without unrelated extras. Enter `0` for an incorrect value,
   omitted requested detail or unrelated extra detail. For `none`, an appropriate uncertainty reply can be correct.
4. Enter your initials in `annotator` and explain mistakes or uncertainty in `notes`.

The letters in `simulator_choice` follow the hidden-detail order: `A` means detail `0`; with two details,
`B` means detail `1` and `C` means none. With one detail, `B` means none. `A,B` can select both details.
Decide your mapping yourself before comparing it with the simulator's choice.

Example: detail `0` is age above 22 and detail `1` is marks above 85. If the question asks only for an
age cutoff, enter `0`. A reply giving only the mark cutoff is incorrect. If the question asks for both
cutoffs, enter `0,1`; a reply giving only one is incomplete.

Use the actual request and hidden values to interpret awkward old slot labels. A poorly phrased question
can still receive a correct simulator reply; question quality and simulator correctness are different checks.
Before committing, check all 247 rows have `human_slot_indices`, `correct` and `annotator` filled.

## 6. Commit and push only your assigned files

Run the commands for **your own role only**, from the repository root. Check the staged file list before
the commit: it should contain only your assigned CSV(s). Avoid `git add .`.

### Reviewer A

```powershell
git add -- annotation/evaluation/question_usefulness_A.csv
git diff --cached --name-only
git commit -m "Complete independent question review A"
git push -u origin review/question-a
```

### Reviewer B

```powershell
git add -- annotation/evaluation/question_usefulness_B.csv
git diff --cached --name-only
git commit -m "Complete independent question review B"
git push -u origin review/question-b
```

### Reviewer C

```powershell
git add -- "annotation/evaluation/simulator_*.csv"
git diff --cached --name-only
git commit -m "Complete simulator review"
git push -u origin review/simulator-c
```

After pushing, send Sathvik your branch name or open a pull request from that branch into `main`.
Do not merge it yourself. Do not inspect other reviews while finishing your own.

## 7. Coordinator: collect, check and score

1. Wait until both independent question reviews are finished before merging review branches into `main`.
2. Keep the original independent ratings and disagreements. Collect seven completed CSVs: two question
   sheets of 120 rows each, plus simulator sheets with 50/50/50/50/47 rows.
3. Check IDs and source columns are unchanged, annotators are distinct for A/B, all judgment fields
   and simulator indices are filled, and no-question items have four zeros. The scorer does not validate
   all of these conditions or establish reviewer independence.
4. Keep the local `question_key.csv` and committed `coverage.json` unchanged. Never publish the key.
   Do not run `make_eval_audits.py` again after review has started.
5. From the repository root in the project Python environment, run:

```powershell
python scripts/score_eval_audits.py
```

The scorer rejects blank binary judgments and unsigned rows. It writes question-usefulness rates,
inter-rater agreement and simulator accuracy to `results/final/analysis/human_audits.json`.
Run it only after all sheets are complete. A fresh clone does not include the private key;
Sathvik's existing local checkout has it for scoring.

6. Add the measured review results to the report and README, rebuild the PDF, and commit the completed
   evaluation. Preserve the frozen benchmark results and original scripted demo score of 5/7.
