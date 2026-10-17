import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import joblib


# --------------------------------
# LOAD PROCESSED DATA
# --------------------------------

df = pd.read_csv("data/processed_returns.csv")

print("Dataset loaded:", df.shape)


# --------------------------------
# FEATURES AND TARGET
# --------------------------------

X = df.drop("Return_Status", axis=1)

y = df["Return_Status"]

print("Input features:", X.shape)
print("Target:", y.shape)


# --------------------------------
# IDENTIFY COLUMN TYPES
# --------------------------------

categorical_features = [
    "Product_Category",
    "Customer_Gender",
    "Payment_Method"
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]


# --------------------------------
# PREPROCESSING PIPELINE
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# --------------------------------
# LOGISTIC REGRESSION MODEL
# --------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# --------------------------------
# TRAIN / TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------
# TRAIN MODEL
# --------------------------------

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------
# PREDICTIONS
# --------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------
# MODEL EVALUATION
# --------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nMODEL PERFORMANCE")
print("-------------------------")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"ROC-AUC Score: {roc_auc:.4f}"
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------
# SAVE MODEL
# --------------------------------

joblib.dump(
    model,
    "models/return_risk_model.pkl"
)

print(
    "\nModel saved successfully at "
    "models/return_risk_model.pkl"
)