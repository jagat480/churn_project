"""
generate_data.py
-----------------
Generates a synthetic customer-churn dataset that mirrors the structure
of the well-known "Telco Customer Churn" dataset (the one on Kaggle).

Why synthetic data? So this project runs end-to-end offline, with no
external downloads. If you have internet access, you can instead download
the real dataset from Kaggle ("Telco Customer Churn" by blastchar) and
save it as data/churn.csv with the same column names — everything else
in this project will work unchanged.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

# --- Demographics ---
gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])

# --- Account info ---
tenure = np.random.randint(0, 73, N)  # months
contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21]
)
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = np.random.choice(
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    N,
)

# --- Services ---
phone_service = np.random.choice(["Yes", "No"], N, p=[0.9, 0.1])
multiple_lines = np.random.choice(["Yes", "No", "No phone service"], N, p=[0.42, 0.48, 0.1])
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])
online_security = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22])
online_backup = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22])
device_protection = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22])
tech_support = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22])
streaming_tv = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.38, 0.40, 0.22])
streaming_movies = np.random.choice(["Yes", "No", "No internet service"], N, p=[0.39, 0.39, 0.22])

# --- Charges ---
monthly_charges = np.round(np.random.normal(64, 30, N).clip(18, 120), 2)
total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, N), 2).clip(0)

# --- Build churn probability from a realistic underlying signal ---
# Month-to-month, fiber optic, high monthly charges, low tenure, no tech support
# all realistically raise churn risk.
risk = (
    0.35 * (contract == "Month-to-month")
    + 0.15 * (internet_service == "Fiber optic")
    + 0.15 * (tech_support == "No")
    + 0.20 * (tenure < 12)
    + 0.10 * (payment_method == "Electronic check")
    + 0.15 * (monthly_charges > 80)
    - 0.20 * (contract == "Two year")
    - 0.10 * (partner == "Yes")
)
prob_churn = 1 / (1 + np.exp(-(risk * 4 - 2.3)))  # squash into a sigmoid, ~27% base churn rate
churn = np.random.binomial(1, prob_churn)

df = pd.DataFrame(
    {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": np.where(churn == 1, "Yes", "No"),
    }
)

import os

os.makedirs("data", exist_ok=True)
df.to_csv("data/churn.csv", index=False)
print(f"Generated {len(df)} rows -> data/churn.csv")
print(df["Churn"].value_counts(normalize=True))
