import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
THRESHOLD_PATH = BASE_DIR / "reports" / "optimal_threshold.json"


# --------------------------------------------------
# 2. LOAD MODEL AND THRESHOLD
# --------------------------------------------------

print("Loading trained ML model...")

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    threshold_data = json.load(file)

threshold = None

for key in ["optimal_threshold", "selected_threshold", "threshold"]:
    if key in threshold_data:
        threshold = float(threshold_data[key])
        break

if threshold is None:
    raise ValueError("Optimal threshold not found.")

print("Model loaded successfully!")
print("Optimal threshold:", threshold)


# --------------------------------------------------
# 3. INITIALIZE FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="Logistics Late Delivery Prediction API",
    description=(
        "Machine Learning API for predicting shipment "
        "late-delivery risk using a trained Logistic Regression model."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# 4. INPUT VALIDATION
# --------------------------------------------------

class ShipmentInput(BaseModel):

    origin_hub: str = Field(min_length=1)
    destination_hub: str = Field(min_length=1)

    route_distance_km: float = Field(gt=0)

    vehicle_type: str = Field(min_length=1)

    traffic_level: str = Field(min_length=1)
    weather_condition: str = Field(min_length=1)

    ambient_temperature_c: float = Field(ge=-50, le=60)

    humidity_percent: float = Field(ge=0, le=100)

    package_temperature_c: float = Field(ge=-50, le=60)

    dispatch_time_utc: datetime

    planned_transit_hours: float = Field(gt=0)

    @field_validator("dispatch_time_utc")
    @classmethod
    def validate_dispatch_timezone(cls, value):

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "dispatch_time_utc must include a timezone, "
                "for example 2025-11-12T10:30:00Z"
            )

        return value.astimezone(timezone.utc)

    @field_validator(
        "origin_hub",
        "destination_hub",
        "vehicle_type",
        "traffic_level",
        "weather_condition"
    )
    @classmethod
    def clean_text_fields(cls, value):

        value = value.strip()

        if not value:
            raise ValueError("Field cannot be empty.")

        return value


# --------------------------------------------------
# 5. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Logistics Late Delivery Prediction API",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Logistic Regression",
        "optimal_threshold": threshold
    }


# --------------------------------------------------
# 6. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_shipment(shipment: ShipmentInput):

    try:

        dispatch_time = shipment.dispatch_time_utc

        # Feature engineering identical to training.
        input_data = {

            "route_distance_km": shipment.route_distance_km,

            "ambient_temperature_c":
                shipment.ambient_temperature_c,

            "humidity_percent":
                shipment.humidity_percent,

            "package_temperature_c":
                shipment.package_temperature_c,

            "planned_transit_hours":
                shipment.planned_transit_hours,

            "dispatch_year": dispatch_time.year,

            "dispatch_month": dispatch_time.month,

            "dispatch_day": dispatch_time.day,

            "dispatch_hour": dispatch_time.hour,

            "dispatch_day_of_week": dispatch_time.weekday(),

            "origin_hub": shipment.origin_hub,

            "destination_hub": shipment.destination_hub,

            "vehicle_type": shipment.vehicle_type,

            "traffic_level": shipment.traffic_level,

            "weather_condition": shipment.weather_condition
        }

        input_df = pd.DataFrame([input_data])

        # Predict probability.
        probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        # Apply optimized threshold.
        prediction = int(probability >= threshold)

        if prediction == 1:
            status = "Late Delivery Risk"
            recommendation = (
                "Flag shipment for proactive monitoring "
                "and possible intervention."
            )
        else:
            status = "Low Late Delivery Risk"
            recommendation = (
                "No immediate late-delivery intervention indicated."
            )

        return {

            "prediction": prediction,

            "status": status,

            "late_delivery_probability": round(
                probability, 4
            ),

            "threshold": threshold,

            "recommendation": recommendation,

            "model": "Logistic Regression",

            "data_note": "Model developed using synthetic shipment data."
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# --------------------------------------------------
# 7. RUN INFORMATION
# --------------------------------------------------

print("\nFastAPI application initialized successfully!")