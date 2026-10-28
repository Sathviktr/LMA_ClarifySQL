# Annotation: what the team does by hand

The proposal (Section 8) asks for human work on the data: test questions are **written or edited by team members**,
two members **label every validation and test episode independently** (we report Cohen's kappa), the **CoSQL dev
slice is cleaned by hand**, and the training data gets a **stratified human audit** in which weak items are dropped.
These CSV files open in Excel or Google Sheets.

To see what is still left at any time:

```
python -I scripts/annotation_status.py
```

## Overview

| Task | Who | Rows | Time | Status |
|---|---|---|---|---|
| 1. Edit the validation and test requests | one editor | 36 + 120 | ~2 h | **done** (SR) |
| 2. Blind labels | two people, A and B, ideally not the editor | 57 + 196 each | ~2 h each | **done** (κ 0.971 / 0.926) |
| 3. Resolve the label disagreements | A and B together | only the rows they disagree on | ~30 min | **done** (8 test dropped) |
| 4. Clean the CoSQL slice | one person | 60 | ~1 h | **done** (59 kept) |
| 5. Audit the training data | one person | 144 | ~1 h | **done** (30 dropped) |

All hand work for Phase 4 is done.

## How to fill a sheet

- Fill only the columns named below, and **sign each row** where the sheet asks for initials. Unsigned rows are
  ignored, so a half-done sheet is safe to hand in.
- Sorting or filtering is fine, but do not delete rows or rename columns.
- **Excel:** open the file, fill it in, then *File → Save As → "CSV UTF-8 (Comma delimited)"* under the **same
  name**.
- **Google Sheets:** *File → Import → Upload*, fill it in, then *File → Download → Comma-separated values*. Rename the
  download back to the exact original name, because Google adds " - Sheet1".
- Put the file back in this `annotation/` folder, replacing the old one, and run the status command above.

## 1. Write and edit requests: `validation_edit.csv`, `test_edit.csv` (done)

Each row is a validation or test episode that is not a plain Spider question. **`final_request`** is the request
the user types and **`editor`** holds the editor's initials. `DROP` removes the episode. All 156 rows are signed: 3
validation and 4 test episodes were dropped, leaving 57 validation and 196 test episodes. `scripts/apply_reviews.py`
applied the edits and re-ran the automatic checks, and every episode passes. The rules used:

1. Start from `original_question` and remove **exactly the detail** in `hidden_detail` (the cut-off value, or the
   measure used for ranking). Keep every other part.
2. It must sound like something a university staff member would type, must not name the hidden measure, and must
   not contain the hidden number.
3. The request must be reasonably interpretable in more than one way, and the hidden intent must be one of them.
4. Definition episodes keep the glossary term as the vague phrase, so that looking it up finds the definition.
5. Unanswerable episodes must be natural, and the database must really not store what is asked.

## 2. Blind labels: `<split>_label_A.csv` and `<split>_label_B.csv` (done)

Both annotators labelled every row. Cohen's kappa between A and B is 0.971 on validation (56 of 57 rows agree) and
0.926 on test (187 of 196); `results/episodes/<split>_agreement.json` has the details.

Annotator A fills `validation_label_A.csv` and `test_label_A.csv`. Annotator B fills the two `_label_B.csv` files.

**Keep it blind.** Open only your own sheets. Do not open `*_label_key.csv`, the files in `data/episodes/`, the
edit sheets or the other annotator's sheets, and do not discuss the rows until both of you are finished. Otherwise
the agreement we report means nothing. The edit sheets reveal the hidden details, which is why the editor should
not be A or B if the team can manage it.

For each row, read the `request`, then look at the `tables` (each line is a table and its columns) and the
`glossary` (definitions the assistant can look up). In the **`decision`** column, enter what a careful assistant
should do **first**:

| decision | when | example |
|---|---|---|
| `execute` | the request is clear enough to write the SQL, possibly after reading a glossary definition | "List the names of countries whose language is not German." / "Which delegates are from counties with small population?" with the glossary entry `small population: population smaller than 100000` |
| `clarify` | an important detail is missing, the glossary does not define it, and only the user can supply it | "Find the policy types used by many customers." (how many is "many"?) / "Which is the best department?" (best by what?) |
| `abstain` | the tables do not store what is asked, so no answer from the user can help | "What is the customer satisfaction score of each airline?" when no table has such a column |

Spend about 20–30 seconds per row. If you are unsure, pick your best guess and write why in `comment`.

## 3. Resolve the disagreements: `<split>_disagreements.csv` (done)

All 15 rows are resolved. The validation row kept its intended decision. 8 of the 14 test rows did not, so those
episodes were dropped, leaving 57 validation and 188 test episodes.

When both annotators have finished, run `python -I scripts/kappa.py validation` (and `test`). It reports Cohen's
kappa and writes `<split>_disagreements.csv`. That sheet lists every row where A and B disagree, or where both
disagree with the episode's `intended` decision. Sit together, and for each row write the right first decision in
**`resolved_decision`** (`execute`, `clarify` or `abstain`), or `drop`. Running `kappa.py` again keeps what you
have already filled in.

If your resolved decision differs from `intended`, the episode is **removed**. For example, both of you may find
that a "vague" request is clear after all. Such an episode is not a clean example of its kind, and the proposal
drops weak items rather than keeping them to reach a target size.

## 4. Clean the CoSQL slice: `cosql_dev_slice.csv` (done)

All 60 turns are cleaned: 59 kept (45 execute, 13 clarify, 1 abstain), 1 dropped. CoSQL's own label matches the
cleaned decision on only 36 of the 59 kept turns: 14 "ambiguous" turns are clear and 9 "cannot answer" turns are
answerable or need a question.

Sixty real user turns from CoSQL dev dialogues on our test databases, with the database's `tables`, the earlier
turns as context and CoSQL's original label. CoSQL labels are noisy: for example, "How many students in this
table?" is labelled CANNOT_ANSWER. Judge each `user_turn` in the light of `earlier_turns`, and use `tables` to
decide whether the data exists. For each row, fill **`cleaned_decision`** with `execute`, `clarify` or `abstain` (as in task 2), or
`drop` if the turn is unusable (a greeting, broken text, or it depends on something not shown). Put your initials
in **`annotator`**. The proposal keeps 40–60 cleaned turns and scores only the agent's first decision on them.

## 5. Audit the training data: `train_audit.csv` (done)

All 144 rows are audited and 30 were dropped, leaving 310 training episodes. The keep rate per stratum is in
`results/episodes/train_audit.json`.

The training requests were made by our rules, by a template or by the base model's paraphrasing. The sheet
contains **every** episode our rules or template wrote (113) plus a fixed sample of the others: 10 plain Spider
questions and 21 model paraphrases. `made_by` says which. Each row shows the original question, the request, the
hidden detail, the glossary entry (definition episodes) and the `reference_path` the model is trained to follow.

Write **`keep`** or **`drop`** in **`verdict`** and your initials in **`auditor`**. Drop the row when:

1. **The request is broken English** that nobody would type, e.g. "Which city does ranks highest?" or "What are the
   names of wrestlers days low held?". Small slips that a real user could make are fine.
2. **ambiguous / two_slot:** the request still contains the hidden detail, it has only one sensible meaning, or
   the question and answer in `reference_path` do not fit the request.
3. **definition:** the glossary term in `definition_in_glossary` does not match the vague phrase in the request,
   e.g. a term like "doe ranks highest".
4. **unanswerable:** the `tables` do store what is asked, or the request is unnatural.
5. **clear:** the request is actually vague, and a careful assistant would have to ask.

Dropping is fine: we do not need to reach a target size. The keep rate per kind is reported as the audit result.

## After the hand work

Run these, or tell whoever maintains the repository that the sheets are in:

```
python -I scripts/annotation_status.py         # everything should say DONE
python -I scripts/kappa.py validation           # and: test (re-run after resolving; it keeps your answers)
python -I scripts/apply_reviews.py validation   # and: test, train
python -I scripts/build_sft.py                  # rebuild the fine-tuning data without the dropped episodes
```

## Files

| File | Filled by | Used for |
|---|---|---|
| `validation_edit.csv`, `test_edit.csv` | editor | final requests |
| `<split>_label_A.csv`, `<split>_label_B.csv` | annotators A and B | Cohen's kappa, final first decisions |
| `<split>_label_key.csv` | (generated, do not open while labelling) | maps the anonymous ids back to episodes |
| `<split>_disagreements.csv` | A and B together | resolved decisions |
| `cosql_dev_slice.csv` | one annotator | the human slice of the evaluation |
| `train_audit.csv` | one auditor | the stratified audit of the training data |

The sheets are made by `scripts/make_review_sheets.py edit|label|cosql|audit`, which refuses to overwrite a sheet
that already exists, so the team's work cannot be lost by accident.
