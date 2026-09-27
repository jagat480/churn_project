# Customer Churn Prediction — End-to-End ML Application

Predicts whether a telecom customer is likely to churn, and serves that
prediction through a REST API with an interactive frontend.

Built to demonstrate: Python, SQL, ML model comparison & evaluation,
model explainability, and application/API development — end to end.

## Project structure

```
churn_project/
├── data/
│   └── churn.csv              # customer dataset (generated or real Telco data)
├── model/
│   ├── churn_model.joblib     # best trained pipeline (preprocessing + classifier)
│   └── feature_schema.joblib  # feature metadata
├── generate_data.py           # creates a synthetic Telco-style dataset
├── sql_eda.py                 # SQL exploratory analysis (SQLite)
├── train_model.py             # trains, compares, and saves the best model
├── app.py                     # FastAPI backend serving /predict
├── frontend.py                # Streamlit UI that calls the API
├── requirements.txt
└── README.md
```

## What it does

1. **Data** — Uses a Telco-style customer dataset (demographics, account
   details, services subscribed, charges). A synthetic generator is
   included so the project runs fully offline; swap in the real
   [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   as `data/churn.csv` for real-world results — the schema matches.
2. **SQL EDA** — `sql_eda.py` loads the data into SQLite and runs queries
   like churn rate by contract type, internet service, and tenure bucket.
3. **Modeling** — `train_model.py` trains and compares Logistic Regression,
   Random Forest, and Gradient Boosting, evaluates them with ROC-AUC,
   precision, recall, and F1, and picks the best performer automatically.
4. **Explainability** — prints the top features driving predictions
   (feature importances / coefficients). A commented SHAP block is
   included if you want deeper, per-prediction explanations.
5. **Serving** — `app.py` exposes a `/predict` FastAPI endpoint that loads
   the saved model and returns a churn probability for a given customer.
6. **UI** — `frontend.py` is a Streamlit form that calls the API and
   displays the churn risk.

## Setup & run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset (skip this if using your own churn.csv)
python generate_data.py

# 3. Explore with SQL (optional)
python sql_eda.py

# 4. Train and save the model
python train_model.py

# 5. Start the API
uvicorn app:app --reload
# -> visit http://127.0.0.1:8000/docs for interactive Swagger docs

# 6. In a separate terminal, start the frontend
streamlit run frontend.py
```

## Results (on synthetic data)

| Model               | ROC-AUC | Precision | Recall | F1   |
|----------------------|---------|-----------|--------|------|
| Random Forest        | ~0.75   | ~0.49     | ~0.74  | ~0.59 |
| Logistic Regression  | ~0.75   | ~0.47     | ~0.76  | ~0.58 |
| Gradient Boosting     | ~0.74   | ~0.55     | ~0.45  | ~0.50 |

Top churn drivers: **month-to-month contracts**, **short tenure**,
**high monthly charges**, and **lack of tech support** — consistent
between the SQL EDA and the model's feature importances.

## Deploying it live

For a shareable link (recommended before an interview):
- **Streamlit Community Cloud** (free) — easiest for the frontend; point it
  at a small FastAPI service or refactor the prediction logic directly
  into the Streamlit app if you want a single-service deploy.
- **Render / Railway** (free tier) — good for hosting the FastAPI backend.
- **Hugging Face Spaces** — supports both Streamlit and FastAPI (via Docker).

## Possible extensions

- Add SHAP for per-customer explanations ("why is *this* customer at risk?")
- Add a `/batch_predict` endpoint that accepts a CSV of customers
- Track experiments with MLflow
- Add authentication + rate limiting to the API
- Containerize with Docker for deployment
