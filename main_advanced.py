import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

DATASET_NAME = "mnemoraorg/telco-churn-7k"


def _target_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() == "churn":
            return c
    raise ValueError("Churn column not found in dataset")


def _to_binary_churn(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.float64).values
    m = s.astype(str).str.strip().str.lower()
    y = m.map({"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    if y.isna().any():
        y = y.fillna(pd.to_numeric(s, errors="coerce"))
    return y.astype(np.float64).values


def decile_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-scores)
    ys = y_true[order]
    n = len(ys)
    rows = []
    for d in range(10):
        lo = d * n // 10
        hi = (d + 1) * n // 10 if d < 9 else n
        seg = ys[lo:hi]
        rows.append({
            "decile": d + 1,
            "n": hi - lo,
            "churn_rate": float(seg.mean()) if len(seg) > 0 else 0.0,
            "churners": int(seg.sum()),
        })
    return pd.DataFrame(rows)


def enrich_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "tenure" not in X.columns:
        return X
    t = X["tenure"].astype(np.float64).clip(lower=1.0)
    if "MonthlyCharges" in X.columns:
        mc = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
        X["MonthlyPerTenure"] = mc / t
    if "TotalCharges" in X.columns:
        tc = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["TotalPerTenure"] = tc / t
    ten = pd.to_numeric(X["tenure"], errors="coerce").fillna(0.0)
    X["TenureLe6"] = (ten <= 6).astype(np.float64)
    X["TenureLe12"] = (ten <= 12).astype(np.float64)
    X["TenureSq"] = ten ** 2
    return X


def load_xy():
    load_dotenv()
    ds = load_dataset(DATASET_NAME, split="train")
    df = ds.to_pandas()
    tcol = _target_column(df)
    y = _to_binary_churn(df[tcol])
    mask = np.isfinite(y)
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]

    X = df.drop(columns=[tcol])
    drop = [c for c in X.columns if c.lower() in {"customerid", "customer id", "id"}]
    if drop:
        X = X.drop(columns=drop)

    if "TotalCharges" in X.columns:
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    X = enrich_features(X)
    return X, y


def main() -> None:
    X, y = load_xy()
    cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    preprocess = ColumnTransformer(transformers)

    base_gb = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("gb", base_gb),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train_i = y_train.astype(np.int32)
    y_test_i = y_test.astype(np.int32)
    sw = compute_sample_weight("balanced", y_train_i)

    search = RandomizedSearchCV(
        pipe,
        param_distributions={
            "gb__n_estimators": randint(100, 400),
            "gb__max_depth": randint(2, 10),
            "gb__learning_rate": uniform(0.02, 0.16),
            "gb__subsample": uniform(0.65, 0.30),
            "gb__min_samples_leaf": randint(1, 28),
            "gb__min_samples_split": randint(2, 24),
        },
        n_iter=32,
        cv=3,
        scoring="average_precision",
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train_i, gb__sample_weight=sw)

    best = search.best_estimator_
    y_score = best.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test_i, y_score)
    pr = average_precision_score(y_test_i, y_score)
    base = float(y_test_i.mean())
    n_test = len(y_test_i)
    k = max(1, n_test // 10)
    top_idx = np.argsort(-y_score)[:k]
    captured_pct = 100.0 * float(y_test_i[top_idx].sum()) / max(1.0, float(y_test_i.sum()))

    print(
        "Tuned GBC + engineered features + balanced sample_weight; "
        "same split as main.py / main_boost.py (random_state=42)"
    )
    print(f"dataset: {DATASET_NAME}")
    print(f"CV best mean PR-AUC: {search.best_score_:.4f}")
    print(f"holdout ROC-AUC: {auc:.4f}")
    print(f"holdout PR-AUC:  {pr:.4f}")
    print(
        f"decile lift (top 10% by risk): capture {captured_pct:.1f}% of test churners; "
        f"churn rate in top decile {float(y_test_i[top_idx].mean()):.4f} vs base {base:.4f}"
    )
    print("decile churn_rate (1= highest predicted risk):")
    dt = decile_table(y_test_i.astype(np.float64), y_score)
    for _, r in dt.iterrows():
        print(f"  {int(r['decile'])}: rate={r['churn_rate']:.4f} n={int(r['n'])}")
    print("best params:")
    for k1, v in sorted(search.best_params_.items()):
        print(f"  {k1}: {v}")

    gb = best.named_steps["gb"]
    names = best.named_steps["prep"].get_feature_names_out()
    imp = gb.feature_importances_
    order = np.argsort(imp)[::-1]
    print("feature_importances_ (top 20):")
    for j in order[:20]:
        if imp[j] <= 0:
            continue
        print(f"  {names[j]}: {imp[j]:.4f}")


if __name__ == "__main__":
    main()
