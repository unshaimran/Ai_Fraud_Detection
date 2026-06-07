import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load datasetimport pandas as pd

df = pd.read_csv("credit_card_fraud_10k.csv")

# remove hidden spaces
df.columns = df.columns.str.strip()

# categorical columns encoding
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# Encoding merchant category (agar text hai)
df["merchant_category"] = df["merchant_category"].astype("category").cat.codes

# Features & Target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

feature_names = X.columns
print(feature_names)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Accuracy check
acc = model.score(X_test, y_test)
print("Model Accuracy:", acc)

# Save model
pickle.dump(model, open("fraud_model.pkl", "wb"))

print("✅ Model Saved Successfully!")

