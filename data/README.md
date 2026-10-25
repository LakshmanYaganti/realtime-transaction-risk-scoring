# Dataset

**Credit Card Fraud Detection** — anonymized credit card transactions made by
European cardholders in September 2013.

| | |
|---|---|
| Rows | 284,807 transactions |
| Fraud | 492 (0.173%) |
| Features | `Time`, `V1`–`V28` (PCA components), `Amount` |
| Target | `Class` (1 = fraud, 0 = legitimate) |

## Provenance

- Original source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
  collected and analysed during a research collaboration between Worldline
  and the Machine Learning Group (MLG) of Université Libre de Bruxelles (ULB).
- Because of the PCA transformation, the original transaction features are
  not recoverable; only `Time` and `Amount` retain their natural meaning.
- **The dataset has no card or merchant identifiers.** The replay harness
  therefore assigns synthetic, seeded card/merchant IDs purely so the
  velocity-enrichment machinery has entities to work over (see
  `src/events.py`). Those IDs are generated, not real.

## Why the CSV is not in this repo

The file is ~150 MB and is distributed under Kaggle's terms, so it is
gitignored (`data/*.csv`). Download it either way:

**Option A — direct mirror (`src/data.py` does this automatically):**

```bash
curl -L -o data/creditcard.csv \
  https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

**Option B — Kaggle:**

Download `creditcard.csv` from the Kaggle dataset page above and place it in
this `data/` directory.

Once `data/creditcard.csv` exists, the training and replay scripts pick it
up automatically (or point `CREDITCARD_DATA_PATH` / `--data` at any other
location).
