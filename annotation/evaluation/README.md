# Checkpoint 7: human review

The model experiments and automatic scoring are complete. These sheets are **blank**, and their judgments
must come from people. The earlier dataset labels and training audit do not replace this evaluation.
The CSVs are generated locally by `scripts/make_eval_audits.py`; keep the system key away from the raters.

## 1. Question usefulness: two independent raters

- Rater A fills `question_usefulness_A.csv`; rater B fills `question_usefulness_B.csv`.
- Each file has **120 items**: the same 40 ambiguous episodes for three systems, shuffled under anonymous IDs.
- Work independently. Do not open `question_key.csv`, the other person's sheet or system-labelled traces
  while rating. The key is only for scoring after both people finish.
- Read the request, intended request, tables and displayed questions/replies. Judge the questions against the
  intended meaning. A bad simulated reply does not automatically make a well-phrased question bad.

Fill each criterion with `1`/`0` or `yes`/`no`:

| Column | Mark yes when |
|---|---|
| `targets_uncertainty` | The question asks about the detail actually missing from the request. |
| `schema_supported` | Its options and assumptions fit the displayed database schema. |
| `not_repeated` | It avoids unnecessary repetition of details already supplied. |
| `resolves_intent` | A cooperative user could provide the missing intent by answering the question(s). |

If the item says no clarification question was asked, mark all four criteria `0` and note this.
If there are several questions, judge the sequence as a whole. Enter your own initials in `annotator`
and explain uncertain cases in `notes`. Preserve IDs and column names. Save as CSV UTF-8.

## 2. Simulator audit

Review the five `simulator_*.csv` files. There are **247 actual decisions**:

| Condition | Decisions to review |
|---|---:|
| B3 | 50 |
| SFT seed 1 | 50 |
| SFT seed 2 | 50 |
| A2 | 50 |
| A3 | 47 |

A3 produced only 47 eligible simulator decisions, so all 47 are included. The final report must disclose
the shortfall from the planned 50; no duplicate or invented examples are added.

For each row, read `request`, `question`, `hidden_details` and the simulator reply. Slot indices start at 0.
Record the indices the natural-language question actually asks about in `human_slot_indices`, or `none`.
Mark `correct` yes/no according to whether the simulator selected and revealed the appropriate details,
then add your initials. Some old slot labels are awkward: use the actual request and hidden values to
interpret them. The question does not need to copy the slot label word for word.

## 3. Score only after the sheets are complete

From the repository root:

```powershell
python scripts/score_eval_audits.py
```

The scorer rejects unsigned or blank judgments. It writes usefulness rates, rater agreement and simulator
accuracy to `results/final/analysis/human_audits.json`. Question rates pool both ratings; keep the independent
sheets and disagreements for the report. The generation script refuses to overwrite existing review work.
