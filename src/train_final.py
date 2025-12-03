# train_final.py
import os, json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import mlflow
from mlflow.models import infer_signature

# 1. Load data
df = pd.read_csv("data/processed/diabetes_processed.csv")
with open("src/selected_features.json", "r") as f:
    selected_features = json.load(f)

X = df[selected_features]
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Load best hyperparameters
with open("models/best_params.json", "r") as f:
    best = json.load(f)

rf_params = best["RandomForest_Grid"]["best_params"]

# 3. Instantiate model
model = RandomForestClassifier(**rf_params, random_state=42)

# 4. Train
model.fit(X_train, y_train)

# 5. Evaluate
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("Final model ROC-AUC on test set:", auc)

# 6. Save model to filesystem
os.makedirs("models/final", exist_ok=True)
model_path = "models/final/model.pkl"
joblib.dump(model, model_path)

# 7. Log to MLflow
mlflow.set_experiment("diabetes_prediction_final")
with mlflow.start_run():
    mlflow.log_params(rf_params)
    mlflow.log_metric("test_roc_auc", float(auc))

    # log model with signature & input example (optional but recommended)
    signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5)
    )

print("Model saved to", model_path)
