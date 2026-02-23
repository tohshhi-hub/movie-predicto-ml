import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Drop missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
