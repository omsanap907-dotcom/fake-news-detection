# Dataset format

Place your permitted CSV dataset here (do not commit large datasets or private data without checking its license and repository visibility).

The training script accepts a CSV with:

- `text`: headline/article text
- `label`: class label, either `FAKE` or `REAL` (case-insensitive)

Example only:

```csv
text,label
"Example claim text",FAKE
"Another example claim",REAL
```

These illustrative rows are not real news examples and must not be used for model training. Add the dataset source and license here before using real data. Aim for both classes to have enough examples. Keep duplicates out of evaluation data, and do not assume the dataset labels are always correct.
