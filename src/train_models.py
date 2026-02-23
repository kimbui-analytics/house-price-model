# src/train_models.py
"""
Train RidgeCV and LassoCV on the processed Ames housing dataset.

What this script demonstrates (resume-friendly):
- Loads a processed dataset artifact (Parquet)
- Builds leakage-safe preprocessing inside sklearn Pipelines (impute + scale)
- Tunes regularization strength via cross-validation (RidgeCV/LassoCV)
- Evaluates on a held-out test set (RMSE)
- Saves a coefficient comparison plot to reports/figures/

Run from repo root:
  python3 src/train_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error


PROCESSED_PATH = Path("data/processed/train_processed.parquet")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    # ----------------------------
    # 1) Load processed dataset
    # ----------------------------
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_PATH}. "
            "Run: python3 src/make_dataset.py"
        )

    df = pd.read_parquet(PROCESSED_PATH)

    # ----------------------------
    # 2) Define target and features
    # ----------------------------
    if "SalePrice" not in df.columns:
        raise ValueError("Expected target column 'SalePrice' not found in processed dataset.")

    y = df["SalePrice"]

    # Drop targets from predictors (ignore if LogSalePrice isn't present for some reason)
    X = df.drop(columns=["SalePrice", "LogSalePrice"], errors="ignore")

    # For this script, keep numeric-only features (simple baseline)
    # Later you can expand to include categorical encoding with ColumnTransformer.
    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        raise ValueError("No numeric features found after filtering. Check your dataset.")

    # ----------------------------
    # 3) Train/test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # 4) Hyperparameter grid
    # ----------------------------
    alphas = np.logspace(-3, 3, 50)

    # ----------------------------
    # 5) RidgeCV pipeline
    #    - Impute missing values with median (leakage-safe inside pipeline)
    #    - Scale features (Ridge/Lasso are scale-sensitive)
    #    - Cross-validate alpha
    # ----------------------------
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=alphas, cv=5)),
    ])

    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    ridge_rmse = rmse(y_test, ridge_preds)
    ridge_alpha = float(ridge.named_steps["model"].alpha_)

    # ----------------------------
    # 6) LassoCV pipeline
    # ----------------------------
    lasso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=42)),
    ])

    lasso.fit(X_train, y_train)
    lasso_preds = lasso.predict(X_test)
    lasso_rmse = rmse(y_test, lasso_preds)
    lasso_alpha = float(lasso.named_steps["model"].alpha_)
    lasso_nonzero = int(np.sum(lasso.named_steps["model"].coef_ != 0))

    # ----------------------------
    # 7) Print results table
    # ----------------------------
    results = pd.DataFrame({
        "Model": ["RidgeCV", "LassoCV"],
        "Best Alpha": [ridge_alpha, lasso_alpha],
        "Test RMSE": [ridge_rmse, lasso_rmse],
        "Non-zero Coefs (Lasso)": [np.nan, lasso_nonzero],
        "Num Features Used": [X.shape[1], X.shape[1]],
    })

    print("\nModel comparison (held-out test set):")
    print(results.to_string(index=False))

    # ----------------------------
    # 8) Save coefficient comparison plot
    #    Note: Coefficients are in standardized feature space due to StandardScaler.
    # ----------------------------
    feature_names = X.columns.tolist()
    ridge_coefs = ridge.named_steps["model"].coef_
    lasso_coefs = lasso.named_steps["model"].coef_

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Ridge": ridge_coefs,
        "Lasso": lasso_coefs,
    })
    coef_df["AbsRidge"] = np.abs(coef_df["Ridge"])

    # Plot top 20 by Ridge magnitude
    top = coef_df.sort_values("AbsRidge", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    plt.plot(top["Feature"], top["Ridge"], marker="o", label="Ridge")
    plt.plot(top["Feature"], top["Lasso"], marker="o", label="Lasso")
    plt.xticks(rotation=90)
    plt.title("Coefficient Comparison: Ridge vs Lasso (Top 20 by Ridge magnitude)")
    plt.legend()
    plt.tight_layout()

    out_fig = FIG_DIR / "ridge_vs_lasso_coefficients_top20.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()

    print(f"\nSaved figure: {out_fig}")


if __name__ == "__main__":
    main()