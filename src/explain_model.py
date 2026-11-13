import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

print("\nLOADING TRAINING DATA, TEST DATA AND MODEL...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["late_delivery"])
X_test = test_df.drop(columns=["late_delivery"])

y_test = test_df["late_delivery"]

model = joblib.load(MODEL_PATH)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 3. EXTRACT PIPELINE COMPONENTS
# --------------------------------------------------

print("\nEXTRACTING PREPROCESSOR AND CLASSIFIER...")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

print("Classifier:", type(classifier).__name__)


# --------------------------------------------------
# 4. TRANSFORM THE DATA
# --------------------------------------------------

print("\nTRANSFORMING DATA...")

X_background = X_train.sample(
    n=min(500, len(X_train)),
    random_state=42
)

X_explain = test_df.sample(
    n=min(500, len(test_df)),
    random_state=42
)

y_explain = X_explain["late_delivery"].copy()

X_explain_features = X_explain.drop(
    columns=["late_delivery"]
)

background_transformed = preprocessor.transform(
    X_background
)

explain_transformed = preprocessor.transform(
    X_explain_features
)

# Convert sparse matrices to dense arrays if required.
if hasattr(background_transformed, "toarray"):
    background_transformed = background_transformed.toarray()

if hasattr(explain_transformed, "toarray"):
    explain_transformed = explain_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

background_df = pd.DataFrame(
    background_transformed,
    columns=feature_names
)

explain_df = pd.DataFrame(
    explain_transformed,
    columns=feature_names
)

print("Original features:", X_train.shape[1])
print("Transformed features:", len(feature_names))


# --------------------------------------------------
# 5. CREATE SHAP EXPLAINER
# --------------------------------------------------

print("\nCREATING SHAP LINEAR EXPLAINER...")

explainer = shap.LinearExplainer(
    classifier,
    background_df
)

print("SHAP explainer created successfully!")


# --------------------------------------------------
# 6. CALCULATE SHAP VALUES
# --------------------------------------------------

print("\nCALCULATING SHAP VALUES...")

shap_values = explainer(explain_df)

print("SHAP values calculated successfully!")
print("Explanation shape:", shap_values.values.shape)


# --------------------------------------------------
# 7. GLOBAL SHAP SUMMARY PLOT
# --------------------------------------------------

print("\nGENERATING SHAP SUMMARY PLOT...")

plt.figure()

shap.plots.beeswarm(
    shap_values,
    max_display=20,
    show=False
)

plt.title("SHAP Summary - Late Delivery Prediction")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 8. SHAP FEATURE IMPORTANCE
# --------------------------------------------------

print("\nGENERATING FEATURE IMPORTANCE PLOT...")

shap.plots.bar(
    shap_values,
    max_display=20,
    show=False
)

plt.title("SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 9. SAVE FEATURE IMPORTANCE CSV
# --------------------------------------------------

print("\nSAVING FEATURE IMPORTANCE...")

importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap_value": np.abs(
        shap_values.values
    ).mean(axis=0)
})

importance_df = importance_df.sort_values(
    by="mean_absolute_shap_value",
    ascending=False
)

importance_df.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTOP 10 IMPORTANT FEATURES:")

print(importance_df.head(10).to_string(index=False))


# --------------------------------------------------
# 10. INDIVIDUAL SHIPMENT EXPLANATION
# --------------------------------------------------

print("\nGENERATING INDIVIDUAL SHIPMENT EXPLANATION...")

# Select a genuinely late shipment from the explanation sample.
late_indices = np.flatnonzero(
    y_explain.to_numpy() == 1
)

if len(late_indices) > 0:
    local_index = int(late_indices[0])
else:
    local_index = 0

local_explanation = shap_values[local_index]

shap.plots.waterfall(
    local_explanation,
    max_display=15,
    show=False
)

plt.title("Individual Shipment SHAP Explanation")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 11. SAVE INDIVIDUAL EXPLANATION CSV
# --------------------------------------------------

individual_df = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values.values[local_index],
    "feature_value": explain_df.iloc[local_index].values
})

individual_df["absolute_shap_value"] = np.abs(
    individual_df["shap_value"]
)

individual_df = individual_df.sort_values(
    by="absolute_shap_value",
    ascending=False
)

individual_df.to_csv(
    REPORT_DIR / "shap_individual_explanation.csv",
    index=False
)

# Save the selected shipment's actual label and predicted probability.
local_probability = model.predict_proba(
    X_explain_features.iloc[[local_index]]
)[0, 1]

print("\nINDIVIDUAL SHIPMENT DETAILS")

print("Actual late delivery:", int(y_explain.iloc[local_index]))
print("Predicted late probability:", round(float(local_probability), 4))


# --------------------------------------------------
# 12. COMPLETION
# --------------------------------------------------

print("\n==========================================")
print("SHAP EXPLAINABILITY COMPLETED!")
print("==========================================")

print("\nSAVED FILES:")

print("reports/figures/shap_summary.png")
print("reports/figures/shap_feature_importance.png")
print("reports/figures/shap_waterfall.png")

print("reports/shap_feature_importance.csv")
print("reports/shap_individual_explanation.csv")

print("\nSHAP ANALYSIS FINISHED!")