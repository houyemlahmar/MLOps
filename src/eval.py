import joblib
import argparse
from sklearn.metrics import accuracy_score
from src.data import load_raw, preprocess, split

def main():
    model = joblib.load("models/model.pkl")
    df = load_raw()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split(X, y)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

