import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_shipments.csv"

REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLOADING CLEANED SHIPMENT DATA...")

df = pd.read_csv(DATA_PATH)

df["dispatch_time_utc"] = pd.to_datetime(
    df["dispatch_time_utc"],
    utc=True
)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. DELIVERY DISTRIBUTION
# --------------------------------------------------

print("\n1. DELIVERY DISTRIBUTION")

print(df["late_delivery"].value_counts())

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="late_delivery"
)

plt.title("Shipment Delivery Distribution")
plt.xlabel("Late Delivery (0 = On-time/Early, 1 = Late)")
plt.ylabel("Number of Shipments")

plt.savefig(
    FIGURE_DIR / "delivery_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 4. LATE DELIVERY BY TRAFFIC
# --------------------------------------------------

print("\n2. LATE DELIVERY RATE BY TRAFFIC")

traffic_analysis = (
    df.groupby("traffic_level", observed=True)["late_delivery"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(traffic_analysis)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=traffic_analysis.index,
    y=traffic_analysis.values
)

plt.title("Late Delivery Rate by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Late Delivery Rate (%)")

plt.savefig(
    FIGURE_DIR / "late_delivery_by_traffic.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 5. LATE DELIVERY BY WEATHER
# --------------------------------------------------

print("\n3. LATE DELIVERY RATE BY WEATHER")

weather_analysis = (
    df.groupby("weather_condition", observed=True)["late_delivery"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(weather_analysis)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=weather_analysis.index,
    y=weather_analysis.values
)

plt.title("Late Delivery Rate by Weather")
plt.xlabel("Weather Condition")
plt.ylabel("Late Delivery Rate (%)")

plt.savefig(
    FIGURE_DIR / "late_delivery_by_weather.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 6. LATE DELIVERY BY VEHICLE
# --------------------------------------------------

print("\n4. LATE DELIVERY RATE BY VEHICLE")

vehicle_analysis = (
    df.groupby("vehicle_type", observed=True)["late_delivery"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(vehicle_analysis)

plt.figure(figsize=(9, 5))

sns.barplot(
    x=vehicle_analysis.index,
    y=vehicle_analysis.values
)

plt.title("Late Delivery Rate by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Late Delivery Rate (%)")

plt.xticks(rotation=15)

plt.savefig(
    FIGURE_DIR / "late_delivery_by_vehicle.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 7. HUB-WISE DELIVERY PERFORMANCE
# --------------------------------------------------

print("\n5. LATE DELIVERY RATE BY ORIGIN HUB")

hub_analysis = (
    df.groupby("origin_hub")["late_delivery"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(hub_analysis.round(2))

plt.figure(figsize=(11, 6))

sns.barplot(
    x=hub_analysis.index,
    y=hub_analysis.values
)

plt.title("Late Delivery Rate by Origin Hub")
plt.xlabel("Origin Hub")
plt.ylabel("Late Delivery Rate (%)")

plt.xticks(rotation=45)

plt.savefig(
    FIGURE_DIR / "late_delivery_by_hub.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 8. NUMERICAL FEATURE CORRELATION
# --------------------------------------------------

print("\n6. NUMERICAL FEATURE CORRELATION")

numeric_columns = [
    "route_distance_km",
    "ambient_temperature_c",
    "humidity_percent",
    "package_temperature_c",
    "planned_transit_hours",
    "late_delivery"
]

correlation = df[numeric_columns].corr()

print(correlation.round(2))

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Numerical Feature Correlation")

plt.savefig(
    FIGURE_DIR / "feature_correlation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 9. MONTHLY SHIPMENT TRENDS
# --------------------------------------------------

print("\n7. MONTHLY SHIPMENT TRENDS")

df["dispatch_month"] = (
    df["dispatch_time_utc"].dt.month
)

monthly_analysis = (
    df.groupby("dispatch_month")
    .agg(
        total_shipments=("shipment_id", "count"),
        late_shipments=("late_delivery", "sum")
    )
)

monthly_analysis["late_delivery_rate"] = (
    monthly_analysis["late_shipments"]
    / monthly_analysis["total_shipments"]
    * 100
)

print(monthly_analysis.round(2))

plt.figure(figsize=(10, 5))

sns.lineplot(
    x=monthly_analysis.index,
    y=monthly_analysis["total_shipments"],
    marker="o"
)

plt.title("Monthly Shipment Volume")
plt.xlabel("Month")
plt.ylabel("Number of Shipments")

plt.xticks(range(1, 13))

plt.savefig(
    FIGURE_DIR / "monthly_shipment_volume.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 10. COMPLETION SUMMARY
# --------------------------------------------------

print("\nEDA COMPLETED SUCCESSFULLY")

print("\nVISUALIZATIONS SAVED IN:")
print(FIGURE_DIR)

print("\nGENERATED FIGURES:")

for file in FIGURE_DIR.glob("*.png"):
    print(file.name)