import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import joblib


# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("data/processed_returns.csv")

X = df.drop("Return_Status", axis=1)
y = df["Return_Status"]


# --------------------------------
# COLUMN TYPES
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
# PREPROCESSOR
# --------------------------------

def create_preprocessor():

    return ColumnTransformer(
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
# TRAIN / TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# MODELS
# --------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
}


# --------------------------------
# TRAIN AND EVALUATE
# --------------------------------

results = []

best_model = None
best_auc = -1
best_model_name = None


for name, classifier in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    })

    if auc > best_auc:

        best_auc = auc
        best_model = pipeline
        best_model_name = name


# --------------------------------
# RESULTS TABLE
# --------------------------------

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="ROC_AUC",
    ascending=False
)

print("\n")
print("=" * 75)

print("MODEL COMPARISON")

print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------
# SAVE RESULTS
# --------------------------------

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# --------------------------------
# SAVE BEST MODEL
# --------------------------------

joblib.dump(
    best_model,
    "models/best_return_risk_model.pkl"
)

print("\nBest Model:", best_model_name)

print(
    "Best ROC-AUC:",
    round(best_auc, 4)
)

print(
    "\nBest model saved at:"
    " models/best_return_risk_model.pkl"
)