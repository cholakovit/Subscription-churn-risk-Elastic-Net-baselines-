import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.compose import ColumnTransformer # used to preprocess the data
from sklearn.impute import SimpleImputer # used to fill in missing values
from sklearn.linear_model import ElasticNetCV # used to train the model
from sklearn.metrics import roc_auc_score, average_precision_score # used to evaluate the model
from sklearn.model_selection import train_test_split # used to split the data into training and testing sets
from sklearn.pipeline import Pipeline # used to create a pipeline of preprocessing and modeling steps
from sklearn.preprocessing import StandardScaler, OneHotEncoder # used to scale the data and one-hot encode the categorical data

DATASET_NAME = "mnemoraorg/telco-churn-7k"

# used to get the target column from the dataset
def _target_column(df: pd.DataFrame) -> str: 
    for c in df.columns:
        if c.lower() == "churn":
            return c
    raise ValueError("Churn column not found in dataset")

# used to convert the churn column to a binary column
def _to_binary_churn(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.float64).values
    m = s.astype(str).str.strip().str.lower()
    y = m.map({"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    if y.isna().any():
        y = y.fillna(pd.to_numeric(s, errors="coerce"))
    return y.astype(np.float64).values

# used to create a decile table
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

def main() -> None:
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

    cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    preprocess = ColumnTransformer(transformers)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv=5,
        random_state=42,
        max_iter=20000,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("enet", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    y_score = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_score)
    pr = average_precision_score(y_test, y_score)
    base = float(y_test.mean())
    n_test = len(y_test)
    k = max(1, n_test // 10)
    top_idx = np.argsort(-y_score)[:k]
    captured_pct = 100.0 * float(y_test[top_idx].sum()) / max(1.0, float(y_test.sum()))

    print(f"dataset: {DATASET_NAME}")
    print(f"holdout ROC-AUC: {auc:.4f}")
    print(f"holdout PR-AUC:  {pr:.4f}")
    print(
        f"decile lift (top 10% by risk): capture {captured_pct:.1f}% of test churners; "
        f"churn rate in top decile {float(y_test[top_idx].mean()):.4f} vs base {base:.4f}"
    )
    print("decile churn_rate (1= highest predicted risk):")
    dt = decile_table(y_test, y_score)
    for _, r in dt.iterrows():
        print(f"  {int(r['decile'])}: rate={r['churn_rate']:.4f} n={int(r['n'])}")

    enet = pipe.named_steps["enet"]
    print(f"ElasticNetCV chosen alpha={enet.alpha_:.6f} l1_ratio={enet.l1_ratio_:.4f}")
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = enet.coef_
    order = np.argsort(np.abs(coef))[::-1]
    print("coefficients (risk score on 0/1 churn; higher score ~ higher churn):")
    for j in order[:25]:
        if abs(coef[j]) < 1e-9:
            continue
        print(f"  {names[j]}: {coef[j]:+.6f}")
    print(f"  intercept: {enet.intercept_:+.6f}")


    print(
        "deployment: batch or online scoring via single dot product after preprocessing; "
        "expect sub-ms per row after one-hot+scale; retrain monthly or on drift alerts; "
        "monitor population stability of top one-hot levels and numeric medians."
    )
    print(
        "ethics: exclude protected attributes from features if present; "
        "audit for leakage (post-churn events, future billing); "
        "use this ranking for outreach prioritization not automated denial."
    )
    
    
if __name__ == "__main__":
    main()