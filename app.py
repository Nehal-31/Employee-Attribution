"""Streamlit dashboard: HR analytics charts, model comparison and individual risk prediction.

Run:  streamlit run app.py
"""
import json
import os
import joblib
import pandas as pd
import streamlit as st

import db
import pipeline as pl

st.set_page_config(page_title="Employee Attrition Risk Platform", layout="wide")
db.init_db()


@st.cache_data
def get_data():
    return pl.load_data()


@st.cache_resource
def get_model(name):
    path = f"models/{name.lower().replace(' ', '_')}.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def get_metrics():
    if not os.path.exists("results/metrics.json"):
        return None
    with open("results/metrics.json") as f:
        return json.load(f)


df = get_data()
metrics = get_metrics()
if metrics is not None:
    # make sure both models are registered even if the database was recreated
    for _n in ["Logistic Regression", "Random Forest"]:
        _m = metrics[_n]
        db.save_model_info(_n, _m["accuracy"], _m["f1"], _m["roc_auc"])
st.title("Employee Attrition Risk Prediction & HR Analytics Platform")

if metrics is None:
    st.error("Models not found. Run `python train.py` first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Attrition Analysis", "Model Results", "Predict Risk"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", len(df))
    c2.metric("Left the company", int(df[pl.TARGET].sum()))
    c3.metric("Attrition rate", f"{df[pl.TARGET].mean() * 100:.1f}%")
    c4.metric("Avg monthly income", f"{df['MonthlyIncome'].mean():,.0f}")
    st.subheader("Data preview")
    st.dataframe(df.head(10), width="stretch")

with tab2:
    st.subheader("Attrition rate by group")
    group = st.selectbox("Group by", ["Department", "OverTime", "JobRole", "MaritalStatus",
                                       "BusinessTravel", "JobSatisfaction", "WorkLifeBalance"])
    rate = (df.groupby(group)[pl.TARGET].mean() * 100).round(1).sort_values()
    st.bar_chart(rate)
    st.caption("Percentage of employees who left, per group.")
    st.subheader("Attrition by age band")
    bands = pd.cut(df["Age"], [17, 25, 35, 45, 55, 70])
    st.bar_chart((df.groupby(bands, observed=True)[pl.TARGET].mean() * 100).round(1)
                 .rename(index=str))

with tab3:
    st.subheader("Model comparison (test set, 20% hold-out)")
    rows = []
    for n in ["Logistic Regression", "Random Forest"]:
        m = metrics[n]
        rows.append({"Model": n, "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"],
                     "CV ROC-AUC": m["cv_roc_auc_mean"]})
    st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
    st.write(f"Best model by F1-score: **{metrics['best_model']}**")
    c1, c2 = st.columns(2)
    if os.path.exists("results/roc_curve.png"):
        c1.image("results/roc_curve.png", caption="ROC curves")
    if os.path.exists("results/feature_importance.png"):
        c2.image("results/feature_importance.png", caption="Feature importance")
    if os.path.exists("results/confusion_matrices.png"):
        st.image("results/confusion_matrices.png", caption="Confusion matrices")

with tab4:
    st.subheader("Predict attrition risk for an employee")
    model_name = st.radio("Model", ["Random Forest", "Logistic Regression"], horizontal=True)
    model = get_model(model_name)
    X, _ = pl.split_xy(df)
    defaults = X.median(numeric_only=True)
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 65, 35)
        income = c1.number_input("Monthly income", 1000, 25000, int(defaults["MonthlyIncome"]))
        years = c1.number_input("Years at company", 0, 40, 5)
        dept = c2.selectbox("Department", sorted(df["Department"].unique()))
        role = c2.selectbox("Job role", sorted(df["JobRole"].unique()))
        overtime = c2.selectbox("Overtime", ["No", "Yes"])
        jobsat = c3.slider("Job satisfaction (1-4)", 1, 4, 3)
        wlb = c3.slider("Work-life balance (1-4)", 1, 4, 3)
        dist = c3.number_input("Distance from home (km)", 1, 30, 8)
        submit = st.form_submit_button("Predict")

    if submit:
        # start from a typical employee, then override the entered fields
        row = pl.default_row(X)
        row.update({"Age": age, "MonthlyIncome": income, "YearsAtCompany": years,
                    "Department": dept, "JobRole": role, "OverTime": overtime,
                    "JobSatisfaction": jobsat, "WorkLifeBalance": wlb, "DistanceFromHome": dist})
        sample = pd.DataFrame([row])[X.columns]
        score = float(model.predict_proba(sample)[0, 1])
        label = pl.risk_label(score)
        st.metric("Attrition risk score", f"{score * 100:.1f}%")
        st.write(f"Risk level: **{label}**")
        st.progress(min(max(score, 0.0), 1.0))
        db.save_prediction(row, model_name, score, label)
        st.success("Prediction saved to the database.")

    st.subheader("Prediction history")
    cols, rows = db.get_history()
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=cols), width="stretch", hide_index=True)
    else:
        st.info("No predictions yet.")
