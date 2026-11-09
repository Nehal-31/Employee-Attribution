"""Data loading, preprocessing and model utilities."""
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/hr_attrition.csv"
TARGET = "Attrition"
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' not found in {path}")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    if df[TARGET].isna().any():
        raise ValueError("Attrition column contains values other than Yes/No")
    return df


def split_xy(df):
    return df.drop(columns=[TARGET]), df[TARGET]


def build_preprocessor(X):
    num = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def build_models(X):
    return {
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor(X)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(X)),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                           min_samples_leaf=3, class_weight="balanced",
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def risk_label(p):
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"


def default_row(X):
    return {c: (float(X[c].median()) if is_numeric_dtype(X[c]) else X[c].mode()[0])
            for c in X.columns}
"""