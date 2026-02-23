import joblib
import pandas as pd
from sklearn.metrics import classification_report
from preprocess import load_and_preprocess

X, y = load_and_preprocess("data/movies.csv")

model = joblib.load("model.pkl")
preds = model.predict(X)

print(classification_report(y, preds))
