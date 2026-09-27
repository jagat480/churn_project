"""
train_model.py
---------------
Trains and compares three classifiers on the customer churn dataset:
Logistic Regression, Random Forest, and Gradient Boosting.

Picks the best model by ROC-AUC, prints a full evaluation report,
shows top feature importances (a simple, dependency-free stand-in for
SHAP — swap in `shap.TreeExplainer` if you have the `shap` package
installed; see the commented block at the bottom), and saves the
winning pipeline to model/churn_model.joblib for the API to load.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------
df = pd.read_csv("data/churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

target = "Churn"
y = (df[target] == "Yes").astype(int)
X = df.drop(columns=[target])

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in X.columns if c not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ---------------------------------------------------------------
# 2. Define candidate models
# ---------------------------------------------------------------
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08, random_state=RANDOM_STATE
    ),
    # If you have xgboost installed, you can swap in:
    # "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
    #                           eval_metric="logloss", random_state=RANDOM_STATE),
}

results = {}
fitted_pipelines = {}

for name, model in candidates.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "roc_auc": auc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
    }
    fitted_pipelines[name] = pipe

    print(f"\n=== {name} ===")
    print(f"ROC-AUC:   {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ---------------------------------------------------------------
# 3. Pick the best model by ROC-AUC
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
print("\n=== Model comparison (sorted by ROC-AUC) ===")
print(results_df.round(4).to_string())

best_name = results_df.index[0]
best_pipeline = fitted_pipelines[best_name]
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------
# 4. Feature importance (explainability)
# ---------------------------------------------------------------
try:
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = best_pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        importances = None

    if importances is not None:
        fi = (
            pd.Series(importances, index=feature_names)
            .sort_values(ascending=False)
            .head(15)
        )
        print("\n=== Top 15 features driving churn predictions ===")
        print(fi.round(4).to_string())
except Exception as e:
    print(f"Could not compute feature importance: {e}")

# ---------------------------------------------------------------
# 5. Save the winning pipeline for the API to load
# ---------------------------------------------------------------
import os

os.makedirs("model", exist_ok=True)
joblib.dump(best_pipeline, "model/churn_model.joblib")
joblib.dump(
    {"numeric_features": numeric_features, "categorical_features": categorical_features},
    "model/feature_schema.joblib",
)
print("\nSaved best pipeline -> model/churn_model.joblib")

# ---------------------------------------------------------------
# Optional: SHAP explainability (uncomment if `pip install shap`)
# ---------------------------------------------------------------
# import shap
# explainer = shap.TreeExplainer(best_pipeline.named_steps["classifier"])
# X_test_transformed = best_pipeline.named_steps["preprocessor"].transform(X_test)
# shap_values = explainer.shap_values(X_test_transformed)
# shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
