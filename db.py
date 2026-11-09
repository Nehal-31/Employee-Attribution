"""SQLite storage for predictions."""
import sqlite3
from datetime import datetime

DB_PATH = "attrition.db"


def get_conn(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path=DB_PATH):
    conn = get_conn(path)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS model (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        accuracy REAL, f1_score REAL, roc_auc REAL);
    CREATE TABLE IF NOT EXISTS employee (
        employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER, department TEXT, job_role TEXT,
        monthly_income REAL, overtime TEXT, years_at_company INTEGER);
    CREATE TABLE IF NOT EXISTS prediction (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL REFERENCES employee(employee_id),
        model_id INTEGER NOT NULL REFERENCES model(model_id),
        risk_score REAL NOT NULL, risk_label TEXT NOT NULL, predicted_on TEXT NOT NULL);
    """)
    conn.commit()
    conn.close()


def save_model_info(name, accuracy, f1, auc, path=DB_PATH):
    conn = get_conn(path)
    conn.execute("""INSERT INTO model(name, accuracy, f1_score, roc_auc) VALUES (?,?,?,?)
                    ON CONFLICT(name) DO UPDATE SET accuracy=excluded.accuracy,
                    f1_score=excluded.f1_score, roc_auc=excluded.roc_auc""",
                 (name, accuracy, f1, auc))
    conn.commit()
    row = conn.execute("SELECT model_id FROM model WHERE name=?", (name,)).fetchone()
    conn.close()
    return row[0]


def save_prediction(emp, model_name, score, label, path=DB_PATH):
    conn = get_conn(path)
    row = conn.execute("SELECT model_id FROM model WHERE name=?", (model_name,)).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Model '{model_name}' is not registered in the database")
    cur = conn.execute(
        "INSERT INTO employee(age, department, job_role, monthly_income, overtime, years_at_company) "
        "VALUES (?,?,?,?,?,?)",
        (int(emp["Age"]), emp["Department"], emp["JobRole"], float(emp["MonthlyIncome"]),
         emp["OverTime"], int(emp["YearsAtCompany"])))
    conn.execute(
        "INSERT INTO prediction(employee_id, model_id, risk_score, risk_label, predicted_on) "
        "VALUES (?,?,?,?,?)",
        (cur.lastrowid, row[0], float(score), label, datetime.now().isoformat(timespec="seconds")))
    conn.commit()
    conn.close()


def get_history(limit=20, path=DB_PATH):
    conn = get_conn(path)
    rows = conn.execute("""
        SELECT p.prediction_id, p.predicted_on, e.age, e.department, e.job_role,
               e.monthly_income, e.overtime, e.years_at_company,
               m.name, p.risk_score, p.risk_label
        FROM prediction p JOIN employee e ON e.employee_id = p.employee_id
        JOIN model m ON m.model_id = p.model_id
        ORDER BY p.prediction_id DESC LIMIT ?""", (limit,)).fetchall()
    conn.close()
    cols = ["ID", "Time", "Age", "Department", "Job Role", "Income", "OverTime",
            "Years", "Model", "Risk Score", "Risk"]
    return cols, rows
"""