import os
import streamlit as st
import requests

from datetime import date, time, datetime, timezone


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Logistics ML Dashboard",
    page_icon="🚚",
    layout="wide"
)


# --------------------------------------------------
# 2. API CONFIGURATION
# --------------------------------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"


# --------------------------------------------------
# 3. CONSTANTS
# --------------------------------------------------

HUBS = [
    "Delhi",
    "Mumbai",
    "Chennai",
    "Bengaluru",
    "Hyderabad",
    "Kolkata",
    "Pune",
    "Jaipur",
    "Lucknow",
    "Ahmedabad",
    "Bhopal",
    "Kochi"
]

VEHICLES = [
    "Refrigerated Van",
    "Reefer Truck",
    "Cold Container"
]

TRAFFIC = [
    "Low",
    "Medium",
    "High"
]

WEATHER = [
    "Clear",
    "Rain",
    "Fog",
    "Storm"
]


# --------------------------------------------------
# 4. DASHBOARD HEADER
# --------------------------------------------------

st.title("🚚 Logistics Late Delivery Prediction")

st.markdown(
    """
    ### AI-Powered Shipment Risk Assessment

    Predict shipment delivery risks using a trained
    Machine Learning model.
    """
)

st.divider()


# --------------------------------------------------
# 5. API STATUS
# --------------------------------------------------

st.subheader("System Status")

try:

    health_response = requests.get(
        HEALTH_URL,
        timeout=3
    )

    if health_response.status_code == 200:

        health_data = health_response.json()

        st.success(
            f"API Online | Model: {health_data['model']} "
            f"| Threshold: {health_data['optimal_threshold']}"
        )

    else:
        st.error("API is not responding correctly.")

except requests.exceptions.RequestException:

    st.error(
        "Cannot connect to FastAPI. "
        "Please start the API server first."
    )

st.divider()


# --------------------------------------------------
# 6. SHIPMENT INPUT FORM
# --------------------------------------------------

st.subheader("📦 Shipment Information")

with st.form("shipment_form"):

    col1, col2 = st.columns(2)

    # ROUTE INFORMATION

    with col1:

        st.markdown("#### Route Details")

        origin_hub = st.selectbox(
            "Origin Hub",
            HUBS,
            index=0
        )

        destination_options = [
            hub for hub in HUBS
            if hub != origin_hub
        ]

        destination_hub = st.selectbox(
            "Destination Hub",
            destination_options
        )

        route_distance = st.number_input(
            "Route Distance (km)",
            min_value=1.0,
            max_value=10000.0,
            value=1400.0
        )

        planned_hours = st.number_input(
            "Planned Transit Hours",
            min_value=0.1,
            max_value=1000.0,
            value=28.0
        )

    # VEHICLE AND ENVIRONMENT

    with col2:

        st.markdown("#### Vehicle & Environment")

        vehicle_type = st.selectbox(
            "Vehicle Type",
            VEHICLES
        )

        traffic_level = st.selectbox(
            "Traffic Level",
            TRAFFIC
        )

        weather_condition = st.selectbox(
            "Weather Condition",
            WEATHER
        )

        ambient_temperature = st.number_input(
            "Ambient Temperature (°C)",
            min_value=-50.0,
            max_value=60.0,
            value=32.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        package_temperature = st.number_input(
            "Package Temperature (°C)",
            min_value=-50.0,
            max_value=60.0,
            value=5.0
        )

    st.divider()

    st.markdown("#### Dispatch Information")

    date_col, time_col = st.columns(2)

    with date_col:

        dispatch_date = st.date_input(
            "Dispatch Date",
            value=date(2025, 11, 12)
        )

    with time_col:

        dispatch_time = st.time_input(
            "Dispatch Time (UTC)",
            value=time(10, 30)
        )

    st.caption(
        "Enter dispatch time in UTC. "
        "Environmental and traffic values represent "
        "information available at dispatch."
    )

    submit = st.form_submit_button(
        "🔍 Predict Delivery Risk",
        use_container_width=True
    )


# --------------------------------------------------
# 7. SEND PREDICTION REQUEST
# --------------------------------------------------

if submit:

    if origin_hub == destination_hub:

        st.error(
            "Origin and destination hubs must be different."
        )

    else:

        dispatch_datetime = datetime.combine(
            dispatch_date,
            dispatch_time
        ).replace(tzinfo=timezone.utc)

        shipment_data = {

            "origin_hub": origin_hub,

            "destination_hub": destination_hub,

            "route_distance_km": route_distance,

            "vehicle_type": vehicle_type,

            "traffic_level": traffic_level,

            "weather_condition": weather_condition,

            "ambient_temperature_c": ambient_temperature,

            "humidity_percent": humidity,

            "package_temperature_c": package_temperature,

            "dispatch_time_utc": dispatch_datetime.isoformat(),

            "planned_transit_hours": planned_hours
        }

        with st.spinner("Analyzing shipment..."):

            try:

                response = requests.post(
                    PREDICT_URL,
                    json=shipment_data,
                    timeout=10
                )

                if response.status_code == 200:

                    result = response.json()

                    st.session_state["prediction_result"] = result

                elif response.status_code == 422:

                    st.error(
                        f"Invalid shipment information: {response.text}"
                    )

                else:

                    st.error(
                        f"API Error {response.status_code}: "
                        f"{response.text}"
                    )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Could not connect to prediction API: {error}"
                )


# --------------------------------------------------
# 8. DISPLAY PREDICTION RESULTS
# --------------------------------------------------

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    st.divider()

    st.header("📊 Prediction Results")

    probability = result["late_delivery_probability"]

    percentage = probability * 100

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Late Delivery Probability",
        f"{percentage:.2f}%"
    )

    metric2.metric(
        "Decision Threshold",
        f"{result['threshold'] * 100:.0f}%"
    )

    metric3.metric(
        "Prediction",
        "Late" if result["prediction"] == 1 else "On-time / Early"
    )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    if result["prediction"] == 1:

        st.error(
            "⚠️ " + result["status"]
        )

    else:

        st.success(
            "✅ " + result["status"]
        )

    st.subheader("Business Recommendation")

    st.info(result["recommendation"])

    st.caption(
        "This is a probabilistic prediction, not a guarantee "
        "of actual delivery performance. The model was developed "
        "using synthetic shipment data."
    )


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Logistics ML Model | "
    "Powered by Scikit-learn, FastAPI and Streamlit"
)