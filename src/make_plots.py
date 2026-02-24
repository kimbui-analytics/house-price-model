# src/make_plots.py
"""
Generate presentation-ready plots for the Ames Housing project.

What this script does:
- Loads processed dataset (Parquet)
- Trains RidgeCV + LassoCV using leakage-safe pipelines (median impute + scale)
- Evaluates on held-out test set
- Saves clean, stakeholder-friendly plots to reports/figures/

Run from repo root:
  python3 src/make_plots.py
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
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def savefig(path: Path) -> None:
    """Save figure consistently for README/Tableau assets."""
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def main() -> None:
    # 1) Load data
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PROCESSED_PATH}. Run: python3 src/make_dataset.py"
        )

    df = pd.read_parquet(PROCESSED_PATH)

    # 2) Define target and features
    if "SalePrice" not in df.columns:
        raise ValueError("Expected target column 'SalePrice' not found.")

    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice", "LogSalePrice"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        raise ValueError("No numeric features found after filtering.")

    feature_names = X.columns.tolist()

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Train RidgeCV + LassoCV
    alphas = np.logspace(-3, 3, 50)

    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=alphas, cv=5)),
    ])

    lasso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=42)),
    ])

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    ridge_preds = ridge.predict(X_test)
    lasso_preds = lasso.predict(X_test)

    ridge_rmse = rmse(y_test, ridge_preds)
    lasso_rmse = rmse(y_test, lasso_preds)

    ridge_alpha = float(ridge.named_steps["model"].alpha_)
    lasso_alpha = float(lasso.named_steps["model"].alpha_)
    lasso_nonzero = int(np.sum(lasso.named_steps["model"].coef_ != 0))

    print("\nModel summary (test set):")
    print(f"RidgeCV  | alpha={ridge_alpha:.6f} | RMSE=${ridge_rmse:,.0f}")
    print(f"LassoCV  | alpha={lasso_alpha:.6f} | RMSE=${lasso_rmse:,.0f} | nonzero_coefs={lasso_nonzero}")

    # 5) Predicted vs Actual (Ridge)
    plt.figure(figsize=(7.5, 6))
    plt.scatter(y_test, ridge_preds, s=20, alpha=0.7)
    lims = [min(y_test.min(), ridge_preds.min()), max(y_test.max(), ridge_preds.max())]
    plt.plot(lims, lims, linewidth=2)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title("Predicted vs Actual Sale Price (RidgeCV)")
    plt.xlabel("Actual SalePrice ($)")
    plt.ylabel("Predicted SalePrice ($)")
    plt.text(0.05, 0.95, f"Test RMSE: ${ridge_rmse:,.0f}", transform=plt.gca().transAxes, verticalalignment="top")
    savefig(FIG_DIR / "predicted_vs_actual_ridge.png")

    # 6) Residuals vs Predicted (Ridge)
    ridge_residuals = y_test - ridge_preds
    plt.figure(figsize=(7.5, 6))
    plt.scatter(ridge_preds, ridge_residuals, s=20, alpha=0.7)
    plt.axhline(0, linewidth=2)
    plt.title("Residuals vs Predicted (RidgeCV)")
    plt.xlabel("Predicted SalePrice ($)")
    plt.ylabel("Residual (Actual - Predicted) ($)")
    savefig(FIG_DIR / "residuals_vs_predicted_ridge.png")

    # 7) Residual distribution (Ridge)
    plt.figure(figsize=(7.5, 6))
    plt.hist(ridge_residuals, bins=40)
    plt.axvline(0, linewidth=2)
    plt.title("Distribution of Prediction Errors (RidgeCV)")
    plt.xlabel("Residual (Actual - Predicted) ($)")
    plt.ylabel("Count")
    savefig(FIG_DIR / "residual_distribution_ridge.png")

    # 8) Top drivers (abs coefficients, Ridge)
    ridge_coefs = ridge.named_steps["model"].coef_
    lasso_coefs = lasso.named_steps["model"].coef_
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "RidgeCoef": ridge_coefs,
        "LassoCoef": lasso_coefs,
        "AbsRidge": np.abs(ridge_coefs),
        "AbsLasso": np.abs(lasso_coefs),
    }).sort_values("AbsRidge", ascending=False)

    top_n = 15
    top = coef_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["Feature"], top["AbsRidge"])
    plt.title(f"Top {top_n} Drivers by Coefficient Magnitude (RidgeCV)")
    plt.xlabel("Absolute Standardized Coefficient")
    savefig(FIG_DIR / "top_drivers_ridge_bar.png")

    # 9) Ridge vs Lasso coefficients (top 20)
    top20 = coef_df.head(20)
    plt.figure(figsize=(11, 6))
    plt.plot(top20["Feature"], top20["RidgeCoef"], marker="o", label="Ridge")
    plt.plot(top20["Feature"], top20["LassoCoef"], marker="o", label="Lasso")
    plt.xticks(rotation=90)
    plt.title("Ridge vs Lasso Coefficients (Top 20 by Ridge magnitude)")
    plt.xlabel("Feature")
    plt.ylabel("Standardized Coefficient")
    plt.legend()
    savefig(FIG_DIR / "ridge_vs_lasso_coefficients_top20.png")

    # 10) Save model results for Tableau
    results = pd.DataFrame({
        "ActualSalePrice": y_test.values,
        "PredSalePrice_Ridge": ridge_preds,
        "PredSalePrice_Lasso": lasso_preds,
    })
    results["Residual_Ridge"] = results["ActualSalePrice"] - results["PredSalePrice_Ridge"]
    results["Residual_Lasso"] = results["ActualSalePrice"] - results["PredSalePrice_Lasso"]
    results["PctError_Ridge"] = results["Residual_Ridge"] / results["ActualSalePrice"]
    results["PctError_Lasso"] = results["Residual_Lasso"] / results["ActualSalePrice"]

    out_csv = Path("data/processed/model_results_testset.csv")
    results.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    print("\nDone. Add these images to README and/or Tableau:")
    print(f"- {FIG_DIR.resolve()}")

if __name__ == "__main__":
    main()
