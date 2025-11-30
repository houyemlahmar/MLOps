import sys, os
import joblib
import mlflow
import argparse
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("diabetes_prediction_experiment")

from src.data import load_raw, preprocess, split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main(params):
    # 1. Load raw data
    df = load_raw("data/raw/diabetes.csv")

    # 2. Preprocess data
    df = preprocess(df)

    # 3. Split into train & test
    X_train, X_test, y_train, y_test = split(df, test_size=params.get('test_size', 0.2))

    # 4. Train model
    model = RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100), 
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. Save model
    joblib.dump(model, "models/model.pkl")

    # 6. Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", params.get('n_estimators', 100))
        mlflow.log_artifact("models/model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    params = vars(args)
    main(params)
