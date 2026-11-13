import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD TRAINING DATA
# --------------------------------------------------

print("\nLOADING TRAINING DATA...")

train_df = pd.read_csv(TRAIN_PATH)

X_train = train_df.drop(columns=["late_delivery"])
y_train = train_df["late_delivery"]

print("Training dataset shape:", train_df.shape)


# --------------------------------------------------
# 3. IDENTIFY FEATURE TYPES
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNUMERICAL FEATURES:")
print(numeric_features)

print("\nCATEGORICAL FEATURES:")
print(categorical_features)


# --------------------------------------------------
# 4. PREPROCESSING PIPELINES
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# --------------------------------------------------
# 5. CLASS IMBALANCE HANDLING
# --------------------------------------------------

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nCLASS DISTRIBUTION:")
print("On-time / early:", negative_count)
print("Late:", positive_count)
print("XGBoost positive class weight:", round(scale_pos_weight, 2))


# --------------------------------------------------
# 6. DEFINE MODELS AND SEARCH SPACES
# --------------------------------------------------

models = {

    "Logistic Regression": {

        "classifier": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        "params": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs", "liblinear"]
        }
    },


    "Decision Tree": {

        "classifier": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "params": {
            "classifier__max_depth": [3, 5, 10, 15, None],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5, 10]
        }
    },


    "Random Forest": {

        "classifier": RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),

        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"]
        }
    },


    "XGBoost": {

        "classifier": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ),

        "params": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0]
        }
    }
}


# --------------------------------------------------
# 7. TIME-AWARE CROSS VALIDATION
# --------------------------------------------------

time_cv = TimeSeriesSplit(n_splits=3)

results = []

best_score = -1
best_model = None
best_model_name = None


# --------------------------------------------------
# 8. TRAIN AND TUNE MODELS
# --------------------------------------------------

for model_name, configuration in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", configuration["classifier"])
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=configuration["params"],
        n_iter=8,
        scoring="average_precision",
        cv=time_cv,
        random_state=42,
        n_jobs=1,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    current_model = search.best_estimator_

    print("\nBEST PARAMETERS:")
    print(search.best_params_)

    print("\nBEST CROSS-VALIDATION PR-AUC:")
    print(round(search.best_score_, 4))

    # Save individual tuned model
    model_filename = (
        model_name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(
        current_model,
        MODEL_DIR / model_filename
    )

    results.append({
        "Model": model_name,
        "Best_CV_PR_AUC": search.best_score_,
        "Best_Parameters": str(search.best_params_)
    })

    if search.best_score_ > best_score:

        best_score = search.best_score_
        best_model = current_model
        best_model_name = model_name


# --------------------------------------------------
# 9. SAVE BEST MODEL
# --------------------------------------------------

print("\n" + "=" * 60)
print("BEST MODEL SELECTION")
print("=" * 60)

print("Best model:", best_model_name)
print("Best CV PR-AUC:", round(best_score, 4))

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib"
)


# --------------------------------------------------
# 10. SAVE MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Best_CV_PR_AUC",
    ascending=False
)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON:")
print(
    results_df[
        ["Model", "Best_CV_PR_AUC"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 11. SAVE TRAINING INFORMATION
# --------------------------------------------------

training_info = {
    "best_model": best_model_name,
    "best_cv_pr_auc": float(best_score),
    "training_records": int(len(X_train)),
    "input_features": int(X_train.shape[1]),
    "randomized_search_iterations": 8,
    "cross_validation_splits": 3,
    "optimization_metric": "Average Precision (PR-AUC)"
}

with open(
    REPORT_DIR / "training_summary.json",
    "w"
) as file:

    json.dump(training_info, file, indent=4)


# --------------------------------------------------
# 12. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSAVED MODELS:")
print("models/logistic_regression.joblib")
print("models/decision_tree.joblib")
print("models/random_forest.joblib")
print("models/xgboost.joblib")
print("models/best_model.joblib")

print("\nSAVED REPORTS:")
print("reports/model_comparison.csv")
print("reports/training_summary.json")