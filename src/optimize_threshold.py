import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. BUSINESS COST ASSUMPTIONS
# --------------------------------------------------

# Illustrative costs, not actual company expenses.

FALSE_NEGATIVE_COST = 1000
FALSE_POSITIVE_COST = 200

print("\nBUSINESS COST ASSUMPTIONS")
print("Missed late shipment cost: Rs.", FALSE_NEGATIVE_COST)
print("Unnecessary intervention cost: Rs.", FALSE_POSITIVE_COST)


# --------------------------------------------------
# 3. LOAD TRAINING DATA AND MODEL
# --------------------------------------------------

print("\nLOADING TRAINING DATA...")

train_df = pd.read_csv(TRAIN_PATH)

X = train_df.drop(columns=["late_delivery"])
y = train_df["late_delivery"]

best_model = joblib.load(MODEL_PATH)

print("Training records:", len(X))


# --------------------------------------------------
# 4. GENERATE OUT-OF-FOLD PROBABILITIES
# --------------------------------------------------

print("\nGENERATING TIME-BASED VALIDATION PREDICTIONS...")

time_cv = TimeSeriesSplit(n_splits=3)

oof_probabilities = np.full(len(X), np.nan)

for fold, (train_idx, val_idx) in enumerate(
    time_cv.split(X), start=1
):

    print(f"\nProcessing validation fold {fold}...")

    fold_model = clone(best_model)

    X_fold_train = X.iloc[train_idx]
    y_fold_train = y.iloc[train_idx]

    X_fold_val = X.iloc[val_idx]

    fold_model.fit(X_fold_train, y_fold_train)

    probabilities = fold_model.predict_proba(
        X_fold_val
    )[:, 1]

    oof_probabilities[val_idx] = probabilities

    print("Training records:", len(train_idx))
    print("Validation records:", len(val_idx))


# --------------------------------------------------
# 5. REMOVE RECORDS WITHOUT OOF PREDICTIONS
# --------------------------------------------------

valid_mask = ~np.isnan(oof_probabilities)

y_validation = y.to_numpy()[valid_mask]
validation_probabilities = oof_probabilities[valid_mask]

print("\nVALIDATION RECORDS:", len(y_validation))

print(
    "Validation PR-AUC:",
    round(
        average_precision_score(
            y_validation,
            validation_probabilities
        ),
        4
    )
)


# --------------------------------------------------
# 6. CALCULATE BUSINESS COST FOR EACH THRESHOLD
# --------------------------------------------------

print("\nOPTIMIZING BUSINESS COST...")

thresholds = np.linspace(0.01, 0.99, 99)

cost_results = []

for threshold in thresholds:

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    false_negatives = np.sum(
        (y_validation == 1) & (predictions == 0)
    )

    false_positives = np.sum(
        (y_validation == 0) & (predictions == 1)
    )

    total_cost = (
        false_negatives * FALSE_NEGATIVE_COST
        +
        false_positives * FALSE_POSITIVE_COST
    )

    cost_results.append({
        "threshold": float(threshold),
        "false_negatives": int(false_negatives),
        "false_positives": int(false_positives),
        "total_cost": int(total_cost),
        "precision": float(
            precision_score(
                y_validation,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_validation,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_validation,
                predictions,
                zero_division=0
            )
        )
    })


# --------------------------------------------------
# 7. SELECT THE LOWEST-COST THRESHOLD
# --------------------------------------------------

cost_df = pd.DataFrame(cost_results)

best_row = cost_df.loc[
    cost_df["total_cost"].idxmin()
]

best_threshold = float(best_row["threshold"])

print("\nOPTIMAL THRESHOLD RESULTS")

print("Selected threshold:", round(best_threshold, 2))
print("Estimated total cost: Rs.", int(best_row["total_cost"]))
print("False negatives:", int(best_row["false_negatives"]))
print("False positives:", int(best_row["false_positives"]))
print("Precision:", round(best_row["precision"], 4))
print("Recall:", round(best_row["recall"], 4))
print("F1-score:", round(best_row["f1_score"], 4))


# --------------------------------------------------
# 8. SAVE THRESHOLD RESULTS
# --------------------------------------------------

cost_df.to_csv(
    REPORT_DIR / "threshold_cost_analysis.csv",
    index=False
)

threshold_information = {
    "selected_threshold": best_threshold,
    "false_negative_cost_inr": FALSE_NEGATIVE_COST,
    "false_positive_cost_inr": FALSE_POSITIVE_COST,
    "estimated_validation_cost_inr": int(
        best_row["total_cost"]
    ),
    "validation_records": int(len(y_validation)),
    "validation_pr_auc": float(
        average_precision_score(
            y_validation,
            validation_probabilities
        )
    ),
    "selection_method": "Minimum estimated business cost",
    "validation_method": "TimeSeriesSplit out-of-fold predictions"
}

with open(
    REPORT_DIR / "optimal_threshold.json",
    "w"
) as file:

    json.dump(
        threshold_information,
        file,
        indent=4
    )


# --------------------------------------------------
# 9. COMPLETION
# --------------------------------------------------

print("\nTHRESHOLD OPTIMIZATION COMPLETED!")

print("\nSAVED FILES:")
print("reports/threshold_cost_analysis.csv")
print("reports/optimal_threshold.json")

print("\nSelected threshold:", round(best_threshold, 2))