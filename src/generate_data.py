import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

SEED = 42
N_SHIPMENTS = 50_000

rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. DEFINE THE 12 FICTIONAL LOGISTICS HUBS
# --------------------------------------------------

hubs = [
    "Delhi", "Mumbai", "Chennai", "Bengaluru",
    "Hyderabad", "Kolkata", "Pune", "Jaipur",
    "Lucknow", "Ahmedabad", "Bhopal", "Kochi"
]

# Some hubs have alternative names in older systems.
hub_aliases = {
    "Delhi": ["Delhi", "New Delhi", "DEL"],
    "Mumbai": ["Mumbai", "Bombay", "MUM"],
    "Chennai": ["Chennai", "Madras", "CHE"],
    "Bengaluru": ["Bengaluru", "Bangalore", "BLR"],
    "Hyderabad": ["Hyderabad", "HYD"],
    "Kolkata": ["Kolkata", "Calcutta", "CCU"],
    "Pune": ["Pune", "PNQ"],
    "Jaipur": ["Jaipur", "JAI"],
    "Lucknow": ["Lucknow", "LKO"],
    "Ahmedabad": ["Ahmedabad", "AMD"],
    "Bhopal": ["Bhopal", "BHO"],
    "Kochi": ["Kochi", "Cochin", "COK"]
}

# --------------------------------------------------
# 3. GENERATE SHIPMENT INFORMATION
# --------------------------------------------------

shipment_ids = [
    f"SHP{number:07d}"
    for number in range(1, N_SHIPMENTS + 1)
]

origin = rng.choice(hubs, N_SHIPMENTS)
destination = rng.choice(hubs, N_SHIPMENTS)

# Prevent origin and destination from being identical.
for i in range(N_SHIPMENTS):
    while destination[i] == origin[i]:
        destination[i] = rng.choice(hubs)

vehicle_types = ["Refrigerated Van", "Reefer Truck", "Cold Container"]

traffic_levels = ["Low", "Medium", "High"]
weather_conditions = ["Clear", "Rain", "Fog", "Storm"]

vehicle = rng.choice(
    vehicle_types,
    N_SHIPMENTS,
    p=[0.40, 0.40, 0.20]
)

traffic = rng.choice(
    traffic_levels,
    N_SHIPMENTS,
    p=[0.30, 0.45, 0.25]
)

weather = rng.choice(
    weather_conditions,
    N_SHIPMENTS,
    p=[0.50, 0.25, 0.15, 0.10]
)

distance = rng.uniform(50, 2500, N_SHIPMENTS)

ambient_temp = rng.uniform(15, 42, N_SHIPMENTS)

humidity = rng.uniform(25, 95, N_SHIPMENTS)

package_temp = rng.normal(4, 2, N_SHIPMENTS)

# --------------------------------------------------
# 4. GENERATE DISPATCH TIMESTAMPS
# --------------------------------------------------

start_date = pd.Timestamp("2025-01-01", tz="UTC")

dispatch_offsets = rng.integers(
    0,
    365 * 24 * 60,
    N_SHIPMENTS
)

dispatch_time = start_date + pd.to_timedelta(
    dispatch_offsets,
    unit="m"
)

# --------------------------------------------------
# 5. GENERATE PLANNED AND ACTUAL DELIVERY TIMES
# --------------------------------------------------

planned_hours = distance / 55 + 2

delay_hours = rng.normal(-3.5, 1.8, N_SHIPMENTS)

# Traffic increases the probability of delays.
delay_hours += np.where(traffic == "High", 2.5, 0)
delay_hours += np.where(traffic == "Medium", 0.8, 0)

# Weather can also increase delays.
delay_hours += np.where(weather == "Rain", 1.2, 0)
delay_hours += np.where(weather == "Fog", 2.0, 0)
delay_hours += np.where(weather == "Storm", 3.0, 0)

# Vehicle condition and package temperature influence delays.
delay_hours += np.where(
    vehicle == "Refrigerated Van", 0.3, 0
)

delay_hours += np.maximum(
    package_temp - 5,
    0
) * 0.25

actual_hours = np.maximum(
    planned_hours + delay_hours,
    0
)

actual_delivery_time = (
    dispatch_time
    + pd.to_timedelta(actual_hours, unit="h")
)

# A shipment is late if actual delivery exceeds planned delivery.
late_delivery = (
    actual_hours > planned_hours
).astype(int)

# --------------------------------------------------
# 6. CREATE THE SHIPMENT DATAFRAME
# --------------------------------------------------

shipments = pd.DataFrame({
    "shipment_id": shipment_ids,
    "origin_hub": origin,
    "destination_hub": destination,
    "route_distance_km": distance.round(2),
    "vehicle_type": vehicle,
    "traffic_level": traffic,
    "weather_condition": weather,
    "ambient_temperature_c": ambient_temp.round(2),
    "humidity_percent": humidity.round(2),
    "package_temperature_c": package_temp.round(2),
    "dispatch_time_utc": dispatch_time,
    "planned_transit_hours": planned_hours.round(2),
    "actual_delivery_time_utc": actual_delivery_time,
    "late_delivery": late_delivery
})

# --------------------------------------------------
# 7. INTRODUCE REALISTIC DATA QUALITY ISSUES
# --------------------------------------------------

# Introduce missing values in selected columns.
for column in [
    "traffic_level",
    "weather_condition",
    "package_temperature_c",
    "humidity_percent"
]:
    missing_indices = rng.choice(
        shipments.index,
        size=int(N_SHIPMENTS * 0.02),
        replace=False
    )

    shipments.loc[missing_indices, column] = np.nan

# Introduce duplicate shipment records.
duplicate_rows = shipments.sample(
    n=100,
    random_state=SEED
)

shipments = pd.concat(
    [shipments, duplicate_rows],
    ignore_index=True
)

# --------------------------------------------------
# 8. GENERATE WAREHOUSE SCAN EVENTS
# --------------------------------------------------

scan_records = []

event_types = [
    "Shipment Dispatched",
    "Arrived at Hub",
    "Quality Check",
    "Out for Delivery"
]

# Different source systems may report timestamps using
# different timezone offsets.
timezone_offsets = ["+00:00", "+05:30", "+01:00", "-04:00"]

for i, row in shipments.iloc[:N_SHIPMENTS].iterrows():

    dispatch = row["dispatch_time_utc"]

    event_offsets = [0.0, 0.20, 0.35, 0.65]

    for event, offset in zip(event_types, event_offsets):

        event_time = (
            dispatch
            + pd.Timedelta(
                hours=row["planned_transit_hours"] * offset
            )
        )

        timezone = rng.choice(timezone_offsets)

        # Convert the event timestamp to the selected timezone.
        timestamp = (
            event_time
            .tz_convert(timezone)
            .isoformat()
        )

        original_hub = (
            row["origin_hub"]
            if event == "Shipment Dispatched"
            else row["destination_hub"]
        )

        # Occasionally use an old or alternative hub name.
        reported_hub = rng.choice(
            hub_aliases[original_hub]
        )

        scan_records.append({
            "shipment_id": row["shipment_id"],
            "event_type": event,
            "reported_hub": reported_hub,
            "scan_timestamp": timestamp
        })

scans = pd.DataFrame(scan_records)

# Shuffle events to simulate out-of-order scan records.
scans = scans.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

# --------------------------------------------------
# 9. SAVE RAW DATASETS
# --------------------------------------------------

shipment_path = RAW_DIR / "shipments.csv"
scan_path = RAW_DIR / "warehouse_scans.csv"

shipments.to_csv(shipment_path, index=False)
scans.to_csv(scan_path, index=False)

# --------------------------------------------------
# 10. DISPLAY GENERATION SUMMARY
# --------------------------------------------------

print("\nSYNTHETIC DATASET GENERATION COMPLETED")
print("--------------------------------------")

print(f"Shipment records: {len(shipments):,}")
print(f"Warehouse scan records: {len(scans):,}")

print(f"\nShipment dataset saved to: {shipment_path}")
print(f"Scan dataset saved to: {scan_path}")

print("\nLate delivery distribution:")
print(shipments["late_delivery"].value_counts())

print("\nShipment dataset preview:")
print(shipments.head())

print("\nMissing values:")
print(shipments.isnull().sum())