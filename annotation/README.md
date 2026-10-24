# Annotation: what the team does by hand

The proposal (Section 8.3) asks for human work on the validation and test data: test questions are
**written or edited by team members**, two members **label every validation and test episode
independently**, and the **CoSQL dev slice is cleaned by hand**. These CSV files open in Excel or Google
Sheets. Edit only the columns named below; do not reorder rows or rename columns.

## 1. Write and edit requests — `validation_edit.csv`, `test_edit.csv` (one editor, about 2 hours)

Each row is a validation or test episode that is not a plain Spider question. Columns to fill:

- **`final_request`**: the request the user types. It is pre-filled with the rule-based draft; empty means
  "write it". Leave it as it is if the draft is fine.
- **`editor`**: your initials. A row is only applied once it is signed here. **`notes`**: anything worth knowing.

Rules for a good vague request:

1. Start from `original_question` and remove **exactly the detail** in `hidden_detail`: the cut-off value
   (threshold) or the measure used for ranking (ranking). Keep every other part: what is listed, other
   conditions, ordering.
2. It must sound like something a university staff member would type. It must not say "ranks highest by
   population" (that names the measure) or contain the hidden number.
3. The request must be **reasonably interpretable in more than one way**, and the hidden intent must be one of
   those interpretations. Example: "Which is the best department?" (by marks, pass rate or placement rate).
4. Definition episodes (`kind = definition`): the vague phrase must match the glossary term, so that looking it
   up finds the definition. Keep the drafted wording of the vague phrase.
5. Unanswerable episodes: check that the request is natural and that the database really cannot answer it.
   Fix the wording if needed.
6. If no sensible vague version exists, write **`DROP`** in `final_request`.

Then run `python -I scripts/apply_reviews.py validation` (and `test`), which applies the edits and re-runs the
automatic checks.

## 2. Blind labels — `<split>_label_A.csv` and `<split>_label_B.csv` (two different people)

Create the sheets **after step 1** with `python -I scripts/make_review_sheets.py label`. Annotator A fills
`_label_A.csv` and annotator B fills `_label_B.csv`, **without talking to each other**. The kind of each
episode is hidden. For each row, read the tables, the glossary and the request, and enter in the `decision`
column what a careful assistant should do **first**:

| decision | when |
|---|---|
| `execute` | the request is clear enough to write the SQL, possibly after reading a glossary definition |
| `clarify` | an important detail is missing and only the user can supply it |
| `abstain` | the database does not store what is asked, so no answer from the user can help |

Then run `python -I scripts/kappa.py validation` (and `test`). It reports Cohen's kappa and writes
`<split>_disagreements.csv`. Discuss each disagreement together, fill `resolved_decision`, and run
`apply_reviews.py` again.

## 3. Clean the CoSQL slice — `cosql_dev_slice.csv` (one person, about 1 hour)

Sixty real user turns from CoSQL dev dialogues on our test databases, with the earlier turns as context and
CoSQL's original label. CoSQL labels are noisy: for example, "How many students in this table?" is labelled
CANNOT_ANSWER. For each row, fill **`cleaned_decision`** with `execute`, `clarify` or `abstain` as defined
above, or `drop` if the turn is unusable (greeting, broken text, depends on something not shown). The proposal
keeps 40–60 cleaned turns and scores only the agent's first decision on them.

## Files

| File | Filled by | Used for |
|---|---|---|
| `validation_edit.csv`, `test_edit.csv` | editor | final requests |
| `<split>_label_A.csv`, `<split>_label_B.csv` | annotators A and B | Cohen's kappa, final first decisions |
| `<split>_label_key.csv` | (generated) | maps the anonymous ids back to episodes |
| `<split>_disagreements.csv` | both annotators together | resolved decisions |
| `cosql_dev_slice.csv` | one annotator | the human slice of the evaluation |
