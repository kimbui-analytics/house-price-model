# src/export_tableau_comparison_dataset_fullfeatures.py
"""
Export a long-format comparison dataset for Tableau that includes:
 - predictions from Ridge and Lasso
 - both Actual and Log-target modeling (predictions back-transformed)
 - residuals, percent error metrics
 - ALL original feature columns (so Tableau can filter / display them)

Saves:
 - data/processed/tableau_model_comparison_fullfeatures.csv
 - data/processed/tableau_model_comparison_fullfeatures.parquet

Run from repo root:
  python3 src/export_tableau_comparison_dataset_fullfeatures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV

PROCESSED_PATH = Path("data/processed/train_processed.parquet")
OUT_CSV = Path("data/processed/tableau_model_comparison_fullfeatures.csv")
OUT_PARQUET = Path("data/processed/tableau_model_comparison_fullfeatures.parquet")

def build_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def main():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_PATH}. Run src/make_dataset.py first.")

    df = pd.read_parquet(PROCESSED_PATH)
    n = len(df)

    # targets
    y_actual = df["SalePrice"].values
    y_log = np.log(y_actual)

    # features used for modeling (numeric baseline)
    X_all = df.drop(columns=["SalePrice", "LogSalePrice"], errors="ignore")
    X_numeric = X_all.select_dtypes(include=[np.number]).copy()
    if X_numeric.shape[1] == 0:
        raise ValueError("No numeric features found to train on. Check processed dataset.")

    # train/test split (we split only to be consistent with modeling practice)
    X_train, X_test, y_train_actual, y_test_actual = train_test_split(
        X_numeric, y_actual, test_size=0.2, random_state=42
    )

    _, _, y_train_log, y_test_log = train_test_split(
        X_numeric, y_log, test_size=0.2, random_state=42
    )

    alphas = np.logspace(-3, 3, 50)
    ridge_pipeline_actual = build_pipeline(RidgeCV(alphas=alphas, cv=5))
    lasso_pipeline_actual = build_pipeline(LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=42))

    # Fit on actual target
    ridge_pipeline_actual.fit(X_train, y_train_actual)
    lasso_pipeline_actual.fit(X_train, y_train_actual)

    # Predict on full numeric set (same ordering as df)
    ridge_preds_actual = ridge_pipeline_actual.predict(X_numeric)
    lasso_preds_actual = lasso_pipeline_actual.predict(X_numeric)

    # Fit on log target (rebuild new pipelines to avoid overwriting)
    ridge_pipeline_log = build_pipeline(RidgeCV(alphas=alphas, cv=5))
    lasso_pipeline_log = build_pipeline(LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=42))

    ridge_pipeline_log.fit(X_train, y_train_log)
    lasso_pipeline_log.fit(X_train, y_train_log)

    # Predict in log space, then back-transform with exp (note: optionally apply bias correction if desired)
    ridge_preds_log = np.exp(ridge_pipeline_log.predict(X_numeric))
    lasso_preds_log = np.exp(lasso_pipeline_log.predict(X_numeric))

    # Build export rows in long format; include all original feature columns
    rows = []
    # Precompute the features as dicts for speed
    feature_dicts = X_all.to_dict(orient="records")  # list of dicts, length n

    for i in range(n):
        base_features = feature_dicts[i]  # dict of all feature columns (excluding SalePrice/LogSalePrice)
        actual = float(y_actual[i])

        # Ridge - Actual target
        pred = float(ridge_preds_actual[i])
        residual = actual - pred
        pct_err = residual / actual if actual != 0 else np.nan
        row = {
            "Model": "Ridge",
            "TargetType": "Actual",
            "SalePrice": actual,
            "PredictedPrice": pred,
            "Residual": residual,
            "PercentError": pct_err,
            "AbsolutePercentError": abs(pct_err) if not np.isnan(pct_err) else np.nan,
        }
        row.update(base_features)
        rows.append(row)

        # Ridge - Log target (back-transformed)
        pred = float(ridge_preds_log[i])
        residual = actual - pred
        pct_err = residual / actual if actual != 0 else np.nan
        row = {
            "Model": "Ridge",
            "TargetType": "Log",
            "SalePrice": actual,
            "PredictedPrice": pred,
            "Residual": residual,
            "PercentError": pct_err,
            "AbsolutePercentError": abs(pct_err) if not np.isnan(pct_err) else np.nan,
        }
        row.update(base_features)
        rows.append(row)

        # Lasso - Actual target
        pred = float(lasso_preds_actual[i])
        residual = actual - pred
        pct_err = residual / actual if actual != 0 else np.nan
        row = {
            "Model": "Lasso",
            "TargetType": "Actual",
            "SalePrice": actual,
            "PredictedPrice": pred,
            "Residual": residual,
            "PercentError": pct_err,
            "AbsolutePercentError": abs(pct_err) if not np.isnan(pct_err) else np.nan,
        }
        row.update(base_features)
        rows.append(row)

        # Lasso - Log target
        pred = float(lasso_preds_log[i])
        residual = actual - pred
        pct_err = residual / actual if actual != 0 else np.nan
        row = {
            "Model": "Lasso",
            "TargetType": "Log",
            "SalePrice": actual,
            "PredictedPrice": pred,
            "Residual": residual,
            "PercentError": pct_err,
            "AbsolutePercentError": abs(pct_err) if not np.isnan(pct_err) else np.nan,
        }
        row.update(base_features)
        rows.append(row)

    export_df = pd.DataFrame(rows)

    # Save Parquet (preferred) + CSV fallback for Tableau if needed
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_parquet(OUT_PARQUET, index=False)
    export_df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size} bytes)")
    print(f"Saved: {OUT_CSV} ({OUT_CSV.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
    