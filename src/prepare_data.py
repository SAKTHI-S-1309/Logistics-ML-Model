import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_shipments.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

print("\nLOADING CLEANED SHIPMENT DATA...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

print("\nCREATING TIME-BASED FEATURES...")

df["dispatch_time_utc"] = pd.to_datetime(
    df["dispatch_time_utc"],
    format="mixed",
    utc=True
)

# Extract information available at dispatch time
df["dispatch_year"] = df["dispatch_time_utc"].dt.year
df["dispatch_month"] = df["dispatch_time_utc"].dt.month
df["dispatch_day"] = df["dispatch_time_utc"].dt.day
df["dispatch_hour"] = df["dispatch_time_utc"].dt.hour
df["dispatch_day_of_week"] = df["dispatch_time_utc"].dt.day_of_week
# Remove the original timestamp after feature extraction
df = df.drop(columns=["dispatch_time_utc"])


# --------------------------------------------------
# 3. REMOVE UNNECESSARY AND LEAKAGE COLUMNS
# --------------------------------------------------

print("\nREMOVING UNNECESSARY COLUMNS...")

columns_to_drop = [
    "shipment_id",
    "actual_delivery_time_utc"
]

df = df.drop(columns=columns_to_drop)


# --------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

print("\nSEPARATING FEATURES AND TARGET...")

X = df.drop(columns=["late_delivery"])
y = df["late_delivery"]

print("Number of input features:", X.shape[1])
print("Target variable: late_delivery")


# --------------------------------------------------
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

print("\nPERFORMING TIME-BASED TRAIN-TEST SPLIT...")

# Sort shipments chronologically using the original
# dispatch timestamp from the cleaned dataset.
original = pd.read_csv(DATA_PATH)

original["dispatch_time_utc"] = pd.to_datetime(
    original["dispatch_time_utc"],
    format="mixed",
    utc=True
)

sorted_indices = original.sort_values(
    "dispatch_time_utc"
).index

X = X.loc[sorted_indices].reset_index(drop=True)
y = y.loc[sorted_indices].reset_index(drop=True)

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()


# --------------------------------------------------
# 6. SAVE TRAINING AND TESTING DATA
# --------------------------------------------------

train_data = X_train.copy()
train_data["late_delivery"] = y_train

test_data = X_test.copy()
test_data["late_delivery"] = y_test

train_data.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

test_data.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\nDATA PREPARATION COMPLETED SUCCESSFULLY!")

print("\nTRAINING DATA SHAPE:", train_data.shape)
print("TESTING DATA SHAPE:", test_data.shape)

print("\nTRAINING TARGET DISTRIBUTION:")
print(y_train.value_counts())

print("\nTESTING TARGET DISTRIBUTION:")
print(y_test.value_counts())

print("\nTRAINING DATA PERIOD:")
print(original.loc[sorted_indices[0], "dispatch_time_utc"])
print("TO")
print(original.loc[sorted_indices[split_index - 1], "dispatch_time_utc"])

print("\nTESTING DATA PERIOD:")
print(original.loc[sorted_indices[split_index], "dispatch_time_utc"])
print("TO")
print(original.loc[sorted_indices[-1], "dispatch_time_utc"])

print("\nSAVED FILES:")
print("data/processed/train.csv")
print("data/processed/test.csv")