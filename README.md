# Employee Attrition Risk Prediction & HR Analytics Platform

Mini Project (Machine Learning), B.Tech CSE, Chandigarh University.
Predicts whether an employee is likely to leave, compares two models, and shows the
results in a Streamlit dashboard. Predictions are stored in a SQLite database.

## Features
- Data cleaning and preprocessing (drop constant/ID columns, one-hot encode, scale)
- Two models: Logistic Regression and Random Forest (class-weight balanced)
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, 5-fold cross-validation, confusion matrix
- Dashboard: overview, attrition analysis charts, model results, individual risk prediction
- SQLite storage of every prediction with its risk score and model used

## Project structure
```
app.py            Streamlit dashboard
train.py          Trains and evaluates the models, saves models + results
pipeline.py       Data loading, preprocessing, model definitions
db.py             SQLite database functions
data/             hr_attrition.csv (IBM HR Analytics Employee Attrition dataset)
models/           Saved models (.pkl), created by train.py
results/          metrics.json and graphs, created by train.py
tests/            Test cases (pytest)
requirements.txt  Dependencies
```

## Installation
Requires Python 3.9 or later.
```
git clone <repository-link>
cd <repository-folder>
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## Execution
```
python train.py                # trains models, writes models/ and results/
streamlit run app.py           # opens the dashboard in the browser
pytest -v                      # runs the test cases
```
Run `python train.py` before the first `streamlit run app.py`.

## Dataset
IBM HR Analytics Employee Attrition & Performance: 1,470 records, 35 columns,
target `Attrition` (about 16% Yes). Fictional data created by IBM data scientists.
Removed columns: EmployeeCount, StandardHours, Over18 (constant) and EmployeeNumber (ID).

## Team
| Member | Contribution |
|---|---|
| D (Team Lead) | Model training and evaluation, pipeline, integration |
| Member 2 | Data cleaning, EDA, metric comparison |
| Member 3 | Streamlit dashboard, SQLite storage, testing, documentation |
