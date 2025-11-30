import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------
# 1. Load raw CSV file
# ----------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ----------------------------------------
# 2. Preprocess data (encode + fill NA)
# ----------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Encode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Fill missing numeric values
    df = df.fillna(df.median(numeric_only=True))

    return df

# ----------------------------------------
# 3. Split features/labels
# ----------------------------------------
def split(df: pd.DataFrame, test_size: float):
    X = df.drop("diabetes", axis=1)   # adjust if label column name differs
    y = df["diabetes"]
    return train_test_split(X, y, test_size=test_size, random_state=42)
