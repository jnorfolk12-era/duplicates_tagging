# About 

This script is an expansion on the original input_data_clean.py script. That script checked for duplicates within the batch. 
It first checks for exact ID matches, then checks cosine similarity between columns of strings (the problems).

This script accepts two csv's as input: 
  one new csv containing problems to be uploaded/delivered, 
  and one csv containing all existing problems, for the new problems to be cross-checked against.

This script returns one csv as output:
  this matches your csv of new problems, but has added columns, 
  primarily a Bool column as to whether or not a problem is a duplicate. (True = duplicate)
  Also shows similarity score and nearest match task_id (for simple auditing).
  When checking for internal duplicates, only the second problem is tagged with True.

For checking for duplicates before uploading Extractions outputs to Solutions:
  download ALL tasks in Solutions, including task data (problem_id is crucial).

For checking for duplicates before auditing/delivering Solutions outputs to customer:
  download tasks in stage "Delivered" in Solutions.


# How to Run in Terminal

```bash
python3 check_pending_against_existing.py \
  --all_csv all_solutions_7-8.csv \
  --pending_csv completed_7-8.csv \
  --all_id_col "problem_id" \
  --all_id_col2 "original_task_id" \
  --all_text_col "problem" \
  --pending_id_col "problem_id" \
  --pending_id_col2 "task_id" \
  --pending_text_col "problem" \
  --out_csv completed_in_solutions_7-8.csv \
  --skip_validation
```

- 'all_csv' -> large csv of existing Solutions problems

- 'pending_csv' -> smaller csv of new problems to be uploaded/delivered

- 'all_id_col' -> columnn containing the problems' unique identifiers in the large csv

- 'all_text_col' -> column containing the text of the problems themselves in the large csv

- 'pending_id_col' -> column containing the problems' unique identifiers in the smaller csv

- 'pending_text_col' -> column containing the text of the problems themselves in the smaller csv

- 'out_csv' -> output file name
  
- 'skip_validation' -> skips over a part of the script that requires an OpenAI code to operate

- 'use_second_id_cols' -> looks at second ID columns. If omitted, only looks at first ID columns.

Note: it may be easiest to keep all relevant files in the same folder as the script when running.
