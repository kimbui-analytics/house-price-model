# src/export_model_coefficients.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV

PROCESSED_PATH = Path("data/processed/train_processed.parquet")
OUT_PATH = Path("data/processed/model_coefficients.csv")

def build_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def main():
    df = pd.read_parquet(PROCESSED_PATH)

    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice", "LogSalePrice"], errors="ignore")
    X_numeric = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42
    )

    alphas = np.logspace(-3, 3, 50)

    ridge = build_pipeline(RidgeCV(alphas=alphas, cv=5))
    lasso = build_pipeline(LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=42))

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    ridge_coefs = ridge.named_steps["model"].coef_
    lasso_coefs = lasso.named_steps["model"].coef_

    coef_df = pd.DataFrame({
    "Feature": X_numeric.columns,
    "Ridge_Coefficient": ridge_coefs,
    "Lasso_Coefficient": lasso_coefs
})

# Compute absolute values
    coef_df["Abs_Ridge"] = coef_df["Ridge_Coefficient"].abs()
    coef_df["Abs_Lasso"] = coef_df["Lasso_Coefficient"].abs()

# Normalize to 0–100 scale (per model)
    max_ridge = coef_df["Abs_Ridge"].max()
    max_lasso = coef_df["Abs_Lasso"].max()

    coef_df["Ridge_Influence_Score"] = (coef_df["Abs_Ridge"] / max_ridge) * 100
    coef_df["Lasso_Influence_Score"] = (coef_df["Abs_Lasso"] / max_lasso) * 100


    coef_df = coef_df.sort_values("Ridge_Influence_Score", ascending=False)

    coef_df.to_csv(OUT_PATH, index=False)
    print(f"Saved coefficients to {OUT_PATH}")

if __name__ == "__main__":
    main()