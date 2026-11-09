"""Test cases for the Employee Attrition project. Run:  pytest -v"""
import json
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db
import pipeline as pl


@pytest.fixture(scope="module")
def df():
    return pl.load_data()


@pytest.fixture(scope="module")
def model():
    import joblib
    return joblib.load("models/random_forest.pkl")


def sample_row(df, **over):
    X, _ = pl.split_xy(df)
    row = pl.default_row(X)
    row.update(over)
    return pd.DataFrame([row])[X.columns]


# ---- Data tests
def test_TC01_dataset_shape(df):
    assert df.shape == (1470, 31)  # 35 columns minus 4 dropped


def test_TC02_constant_and_id_columns_removed(df):
    for c in pl.DROP_COLS:
        assert c not in df.columns


def test_TC03_no_missing_values(df):
    assert df.isna().sum().sum() == 0


def test_TC04_target_is_binary(df):
    assert set(df[pl.TARGET].unique()) == {0, 1}


def test_TC05_attrition_rate_about_16_percent(df):
    assert 0.15 < df[pl.TARGET].mean() < 0.17


def test_TC06_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pl.load_data("data/does_not_exist.csv")


def test_TC07_bad_file_without_target_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        pl.load_data(str(p))


# ---- Model tests
def test_TC08_probability_between_0_and_1(df, model):
    p = model.predict_proba(sample_row(df))[0, 1]
    assert 0.0 <= p <= 1.0


def test_TC09_high_risk_profile_scores_above_low_risk_profile(df, model):
    high = sample_row(df, OverTime="Yes", JobSatisfaction=1, WorkLifeBalance=1,
                      MonthlyIncome=2000, Age=22, YearsAtCompany=1)
    low = sample_row(df, OverTime="No", JobSatisfaction=4, WorkLifeBalance=4,
                     MonthlyIncome=15000, Age=45, YearsAtCompany=15)
    assert model.predict_proba(high)[0, 1] > model.predict_proba(low)[0, 1]


def test_TC10_unseen_category_does_not_crash(df, model):
    row = sample_row(df, Department="Unknown Dept")
    p = model.predict_proba(row)[0, 1]
    assert 0.0 <= p <= 1.0


def test_TC11_extreme_values_do_not_crash(df, model):
    row = sample_row(df, MonthlyIncome=10**7, Age=100, YearsAtCompany=80)
    assert 0.0 <= model.predict_proba(row)[0, 1] <= 1.0


def test_TC12_model_beats_always_no_baseline_on_recall():
    m = json.load(open("results/metrics.json"))
    assert m["Random Forest"]["recall"] > 0.0
    assert m["Logistic Regression"]["recall"] > 0.0


def test_TC13_roc_auc_above_random():
    m = json.load(open("results/metrics.json"))
    assert m["Random Forest"]["roc_auc"] > 0.7
    assert m["Logistic Regression"]["roc_auc"] > 0.7


@pytest.mark.parametrize("p,label", [(0.0, "Low"), (0.29, "Low"), (0.30, "Medium"),
                                     (0.59, "Medium"), (0.60, "High"), (1.0, "High")])
def test_TC14_risk_label_boundaries(p, label):
    assert pl.risk_label(p) == label


# ---- Database tests
EMP = {"Age": 30, "Department": "Sales", "JobRole": "Sales Executive",
       "MonthlyIncome": 5000, "OverTime": "Yes", "YearsAtCompany": 3}


def test_TC15_db_save_and_read_prediction(tmp_path):
    path = str(tmp_path / "t.db")
    db.init_db(path)
    db.save_model_info("Random Forest", 0.8, 0.5, 0.8, path)
    db.save_prediction(EMP, "Random Forest", 0.42, "Medium", path)
    cols, rows = db.get_history(10, path)
    assert len(rows) == 1 and rows[0][-1] == "Medium"


def test_TC16_db_unregistered_model_rejected(tmp_path):
    path = str(tmp_path / "t.db")
    db.init_db(path)
    with pytest.raises(ValueError):
        db.save_prediction(EMP, "Nonexistent Model", 0.5, "Medium", path)


def test_TC17_db_history_empty_initially(tmp_path):
    path = str(tmp_path / "t.db")
    db.init_db(path)
    assert db.get_history(10, path)[1] == []


def test_TC18_db_foreign_key_enforced(tmp_path):
    import sqlite3
    path = str(tmp_path / "t.db")
    db.init_db(path)
    conn = db.get_conn(path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO prediction(employee_id, model_id, risk_score, risk_label, predicted_on)"
                     " VALUES (999, 999, 0.5, 'Medium', 'x')")
    conn.close()


def test_TC19_db_model_metrics_upsert(tmp_path):
    path = str(tmp_path / "t.db")
    db.init_db(path)
    a = db.save_model_info("Random Forest", 0.8, 0.5, 0.8, path)
    b = db.save_model_info("Random Forest", 0.9, 0.6, 0.9, path)
    assert a == b


def test_TC20_models_can_be_registered_from_metrics_on_fresh_db(tmp_path):
    """Regression test: a freshly created database must accept predictions after
    registering models from results/metrics.json (as app.py does at startup)."""
    path = str(tmp_path / "fresh.db")
    db.init_db(path)
    m = json.load(open("results/metrics.json"))
    for n in ["Logistic Regression", "Random Forest"]:
        db.save_model_info(n, m[n]["accuracy"], m[n]["f1"], m[n]["roc_auc"], path)
    db.save_prediction(EMP, "Random Forest", 0.5, "Medium", path)
    assert len(db.get_history(10, path)[1]) == 1
