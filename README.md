# Subscription churn risk (Elastic Net + baselines)

## Business brief

A B2B SaaS team wants a **transparent baseline** to **rank accounts by churn risk** so Customer Success can prioritize outreach. Leadership cares about **interpretability** (coefficients), **stability** with correlated usage fields, and standard **ranking metrics** on a holdout set.

## Dataset

**[mnemoraorg/telco-churn-7k](https://huggingface.co/datasets/mnemoraorg/telco-churn-7k)** — tabular Telco-style churn, loaded with `datasets.load_dataset`. Optional: set `HF_TOKEN` in `.env` for Hub rate limits (`python-dotenv`).

## Models in this repo

| Script | Role |
|--------|------|
| `main.py` | **ElasticNetCV** on a 0/1 churn target (risk score), numeric scaling + one-hot categoricals, internal CV for `alpha` / `l1_ratio`. |
| `main_boost.py` | **GradientBoostingClassifier** on the same **80/20 split** (`random_state=42`, stratified) for comparison. |
| `main_advanced.py` | Tuned **GradientBoostingClassifier** (RandomizedSearchCV on PR-AUC), **engineered numeric features**, **balanced `sample_weight`** on training; same split as above. |

All three scripts report **holdout ROC-AUC**, **PR-AUC**, **decile churn rates**, and **top-decile share of churners**. Elastic Net prints **signed coefficients** (with preprocessor feature names); boosters print **`feature_importances_`**. `main.py` also prints short **deployment** and **ethics** lines aligned with the original brief.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python main.py
uv run python main_boost.py
uv run python main_advanced.py
```

Metrics are comparable across scripts because the **test split is identical**.

## Original deliverables (brief)

- **ROC-AUC / PR-AUC** on a holdout + **decile lift** (e.g. share of churners in the top 10% of predicted risk).
- **Coefficient report** (Elastic Net) mapped to business drivers — signs/magnitudes are associative, not causal.
- **Deployment:** low-latency scoring after a fixed preprocessing + linear weights; retrain on a cadence or on drift; monitor input distributions.
- **Ethics / compliance:** omit sensitive attributes from features; check for **leakage** (post-churn signals, future billing); use rankings to **prioritize outreach**, not as a sole automated denial.
