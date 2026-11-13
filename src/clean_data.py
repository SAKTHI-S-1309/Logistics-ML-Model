import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD RAW DATASETS
# --------------------------------------------------

print("\nLOADING RAW DATASETS...")

shipments = pd.read_csv(RAW_DIR / "shipments.csv")
scans = pd.read_csv(RAW_DIR / "warehouse_scans.csv")

print("Original shipment records:", len(shipments))
print("Original scan records:", len(scans))

# --------------------------------------------------
# 3. REMOVE DUPLICATE SHIPMENTS
# --------------------------------------------------

print("\nREMOVING DUPLICATE SHIPMENTS...")

shipments = shipments.drop_duplicates(
    subset="shipment_id",
    keep="first"
).copy()

print("Shipment records after duplicate removal:", len(shipments))

# --------------------------------------------------
# 4. STANDARDIZE HUB NAMES
# --------------------------------------------------

hub_mapping = {
    "Delhi": "Delhi",
    "New Delhi": "Delhi",
    "DEL": "Delhi",

    "Mumbai": "Mumbai",
    "Bombay": "Mumbai",
    "MUM": "Mumbai",

    "Chennai": "Chennai",
    "Madras": "Chennai",
    "CHE": "Chennai",

    "Bengaluru": "Bengaluru",
    "Bangalore": "Bengaluru",
    "BLR": "Bengaluru",

    "Hyderabad": "Hyderabad",
    "HYD": "Hyderabad",

    "Kolkata": "Kolkata",
    "Calcutta": "Kolkata",
    "CCU": "Kolkata",

    "Pune": "Pune",
    "PNQ": "Pune",

    "Jaipur": "Jaipur",
    "JAI": "Jaipur",

    "Lucknow": "Lucknow",
    "LKO": "Lucknow",

    "Ahmedabad": "Ahmedabad",
    "AMD": "Ahmedabad",

    "Bhopal": "Bhopal",
    "BHO": "Bhopal",

    "Kochi": "Kochi",
    "Cochin": "Kochi",
    "COK": "Kochi"
}

shipments["origin_hub"] = (
    shipments["origin_hub"].str.strip().map(hub_mapping)
)

shipments["destination_hub"] = (
    shipments["destination_hub"].str.strip().map(hub_mapping)
)

scans["reported_hub"] = (
    scans["reported_hub"].str.strip().map(hub_mapping)
)

print("\nHUB NAMES STANDARDIZED.")

# Check for unknown hub names.
unknown_origin = shipments["origin_hub"].isna().sum()
unknown_destination = shipments["destination_hub"].isna().sum()
unknown_scan_hubs = scans["reported_hub"].isna().sum()

print("Unknown origin hubs:", unknown_origin)
print("Unknown destination hubs:", unknown_destination)
print("Unknown scan hubs:", unknown_scan_hubs)

# --------------------------------------------------
# 5. NORMALIZE TIMESTAMPS TO UTC
# --------------------------------------------------

print("\nNORMALIZING TIMESTAMPS...")

shipments["dispatch_time_utc"] = pd.to_datetime(
    shipments["dispatch_time_utc"],
    format="mixed",
    utc=True,
    errors="coerce"
)

shipments["actual_delivery_time_utc"] = pd.to_datetime(
    shipments["actual_delivery_time_utc"],
    format="mixed",
    utc=True,
    errors="coerce"
)

scans["scan_timestamp_utc"] = pd.to_datetime(
    scans["scan_timestamp"],
    format="mixed",
    utc=True,
    errors="coerce"
)

print(
    "Invalid shipment dispatch timestamps:",
    shipments["dispatch_time_utc"].isna().sum()
)

print(
    "Invalid delivery timestamps:",
    shipments["actual_delivery_time_utc"].isna().sum()
)

print(
    "Invalid scan timestamps:",
    scans["scan_timestamp_utc"].isna().sum()
)

# --------------------------------------------------
# 6. SORT WAREHOUSE SCAN EVENTS
# --------------------------------------------------

print("\nSORTING WAREHOUSE EVENTS...")

scans = scans.sort_values(
    by=["shipment_id", "scan_timestamp_utc"]
).reset_index(drop=True)

# Assign chronological event sequence per shipment.
scans["event_sequence"] = (
    scans.groupby("shipment_id").cumcount() + 1
)

print("Warehouse events sorted chronologically.")

# --------------------------------------------------
# 7. VALIDATE CLEANED DATA
# --------------------------------------------------

print("\nVALIDATING CLEANED DATA...")

assert shipments["shipment_id"].is_unique

assert shipments["origin_hub"].notna().all()
assert shipments["destination_hub"].notna().all()

assert scans["reported_hub"].notna().all()

assert shipments["dispatch_time_utc"].notna().all()
assert shipments["actual_delivery_time_utc"].notna().all()

assert scans["scan_timestamp_utc"].notna().all()

# --------------------------------------------------
# 8. SAVE PROCESSED DATASETS
# --------------------------------------------------

shipment_output = PROCESSED_DIR / "cleaned_shipments.csv"
scan_output = PROCESSED_DIR / "cleaned_warehouse_scans.csv"

shipments.to_csv(shipment_output, index=False)
scans.to_csv(scan_output, index=False)

# --------------------------------------------------
# 9. DISPLAY SUMMARY
# --------------------------------------------------

print("\nDATA CLEANING COMPLETED SUCCESSFULLY")
print("-------------------------------------")

print("Cleaned shipment records:", len(shipments))
print("Cleaned scan records:", len(scans))

print("\nShipment output:", shipment_output)
print("Scan output:", scan_output)

print("\nLate delivery distribution:")
print(shipments["late_delivery"].value_counts())

print("\nRemaining missing values:")
print(shipments.isnull().sum())

print("\nFIRST FIVE CHRONOLOGICALLY SORTED SCAN EVENTS:")
print(
    scans[
        [
            "shipment_id",
            "event_type",
            "reported_hub",
            "scan_timestamp_utc",
            "event_sequence"
        ]
    ].head()
)