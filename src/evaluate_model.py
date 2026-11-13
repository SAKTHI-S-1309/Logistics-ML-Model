import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
THRESHOLD_PATH = BASE_DIR / "reports" / "optimal_threshold.json"

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. BUSINESS COST ASSUMPTIONS
# --------------------------------------------------

FN_COST = 1000
FP_COST = 200

print("\nBUSINESS COST ASSUMPTIONS")
print("Missed late shipment cost: Rs.", FN_COST)
print("Unnecessary intervention cost: Rs.", FP_COST)


# --------------------------------------------------
# 3. LOAD DATA AND MODEL
# --------------------------------------------------

print("\nLOADING TEST DATA AND BEST MODEL...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["late_delivery"])
y_train = train_df["late_delivery"]

X_test = test_df.drop(columns=["late_delivery"])
y_test = test_df["late_delivery"]

best_model = joblib.load(MODEL_PATH)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 4. LOAD OPTIMAL THRESHOLD
# --------------------------------------------------

print("\nLOADING OPTIMAL THRESHOLD...")

with open(THRESHOLD_PATH, "r") as file:
    threshold_data = json.load(file)

# Support common threshold key names.
possible_keys = [
    "optimal_threshold",
    "selected_threshold",
    "threshold"
]

optimal_threshold = None

for key in possible_keys:
    if key in threshold_data:
        optimal_threshold = float(threshold_data[key])
        break

if optimal_threshold is None:
    raise ValueError(
        "Could not find the selected threshold in optimal_threshold.json. "
        "Please check the JSON file."
    )

print("Optimized threshold:", optimal_threshold)


# --------------------------------------------------
# 5. GENERATE PROBABILITIES
# --------------------------------------------------

print("\nGENERATING TEST PREDICTIONS...")

y_prob = best_model.predict_proba(X_test)[:, 1]

thresholds = {
    "Default_0.50": 0.50,
    "Optimized": optimal_threshold
}


# --------------------------------------------------
# 6. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_threshold(name, threshold):

    predictions = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    total_cost = (fn * FN_COST) + (fp * FP_COST)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "estimated_business_cost": int(total_cost)
    }

    return predictions, metrics


# --------------------------------------------------
# 7. COMPARE BOTH THRESHOLDS
# --------------------------------------------------

print("\nEVALUATING BOTH THRESHOLDS...")

all_results = {}
all_predictions = {}

for name, threshold in thresholds.items():

    predictions, metrics = evaluate_threshold(
        name,
        threshold
    )

    all_results[name] = metrics
    all_predictions[name] = predictions

    print("\n----------------------------------")
    print(name)
    print("----------------------------------")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")


# --------------------------------------------------
# 8. BASELINE COMPARISON
# --------------------------------------------------

print("\nCOMPARING WITH BASELINE MODEL...")

baseline = DummyClassifier(strategy="prior")

baseline.fit(X_train, y_train)

baseline_prob = baseline.predict_proba(X_test)[:, 1]

baseline_pr_auc = average_precision_score(
    y_test,
    baseline_prob
)

model_pr_auc = all_results["Optimized"]["pr_auc"]

improvement = (
    (model_pr_auc - baseline_pr_auc)
    / baseline_pr_auc
) * 100

print("Baseline PR-AUC:", round(baseline_pr_auc, 4))
print("Model PR-AUC:", round(model_pr_auc, 4))
print("Improvement over baseline:", round(improvement, 2), "%")


# --------------------------------------------------
# 9. CONFUSION MATRICES
# --------------------------------------------------

print("\nGENERATING CONFUSION MATRICES...")

for name, predictions in all_predictions.items():

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["On-time / Early", "Late"]
    )

    display.plot()

    plt.title(f"Late Delivery Prediction - {name}")
    plt.tight_layout()

    filename = (
        "confusion_matrix_optimized.png"
        if name == "Optimized"
        else "confusion_matrix_default.png"
    )

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300
    )

    plt.close()

# Keep the original confusion matrix filename for the optimized model.
import shutil

shutil.copy2(
    FIGURE_DIR / "confusion_matrix_optimized.png",
    FIGURE_DIR / "confusion_matrix.png"
)


# --------------------------------------------------
# 10. CLASSIFICATION REPORTS
# --------------------------------------------------

print("\nCLASSIFICATION REPORTS")

for name, predictions in all_predictions.items():

    print("\n", name)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["On-time / Early", "Late"],
            zero_division=0
        )
    )


# --------------------------------------------------
# 11. SAVE COMPARISON RESULTS
# --------------------------------------------------

comparison_df = pd.DataFrame(all_results).T

comparison_df.to_csv(
    REPORT_DIR / "threshold_evaluation_comparison.csv"
)


# --------------------------------------------------
# 12. SAVE FINAL JSON RESULTS
# --------------------------------------------------

results = {
    "model": "Logistic Regression",
    "test_records": int(len(X_test)),
    "business_cost_assumptions": {
        "false_negative_cost": FN_COST,
        "false_positive_cost": FP_COST
    },
    "baseline_pr_auc": float(baseline_pr_auc),
    "improvement_over_baseline_percent": float(improvement),
    "threshold_comparison": all_results
}

with open(
    REPORT_DIR / "evaluation_results.json",
    "w"
) as file:

    json.dump(results, file, indent=4)


# --------------------------------------------------
# 13. SAVE TEST PREDICTIONS
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "actual_late_delivery": y_test.values,
    "late_delivery_probability": y_prob,
    "prediction_at_0.50": all_predictions["Default_0.50"],
    "prediction_at_optimized_threshold": all_predictions["Optimized"]
})

prediction_results.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False
)


# --------------------------------------------------
# 14. FINAL SUMMARY
# --------------------------------------------------

print("\n==========================================")
print("FINAL EVALUATION COMPLETED SUCCESSFULLY!")
print("==========================================")

print("\nBUSINESS COST COMPARISON")

for name, metrics in all_results.items():

    print(
        f"{name}: Rs. {metrics['estimated_business_cost']:,}"
    )

print("\nSAVED FILES:")

print("reports/evaluation_results.json")
print("reports/threshold_evaluation_comparison.csv")
print("reports/test_predictions.csv")

print("reports/figures/confusion_matrix.png")
print("reports/figures/confusion_matrix_default.png")
print("reports/figures/confusion_matrix_optimized.png")

print("\nEVALUATION FINISHED!")
