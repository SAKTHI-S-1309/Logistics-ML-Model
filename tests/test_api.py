from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# Test 1: Check root endpoint
def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200


# Test 2: Check API health
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "Logistic Regression"
    assert data["optimal_threshold"] == 0.33


# Test 3: Check valid shipment prediction
def test_valid_shipment_prediction():

    shipment = {
        "origin_hub": "Delhi",
        "destination_hub": "Mumbai",
        "route_distance_km": 1400,
        "vehicle_type": "Reefer Truck",
        "traffic_level": "High",
        "weather_condition": "Rain",
        "ambient_temperature_c": 32,
        "humidity_percent": 75,
        "package_temperature_c": 5,
        "dispatch_time_utc": "2025-11-12T10:30:00Z",
        "planned_transit_hours": 28
    }

    response = client.post("/predict", json=shipment)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0 <= data["late_delivery_probability"] <= 1
    assert data["optimal_threshold"] if "optimal_threshold" in data else data["threshold"] == 0.33
    assert "status" in data
    assert "recommendation" in data


# Test 4: Reject invalid humidity
def test_invalid_humidity():

    shipment = {
        "origin_hub": "Delhi",
        "destination_hub": "Mumbai",
        "route_distance_km": 1400,
        "vehicle_type": "Reefer Truck",
        "traffic_level": "High",
        "weather_condition": "Rain",
        "ambient_temperature_c": 32,
        "humidity_percent": 120,
        "package_temperature_c": 5,
        "dispatch_time_utc": "2025-11-12T10:30:00Z",
        "planned_transit_hours": 28
    }

    response = client.post("/predict", json=shipment)

    assert response.status_code == 422


# Test 5: Reject dispatch time without timezone
def test_invalid_dispatch_timezone():

    shipment = {
        "origin_hub": "Delhi",
        "destination_hub": "Mumbai",
        "route_distance_km": 1400,
        "vehicle_type": "Reefer Truck",
        "traffic_level": "High",
        "weather_condition": "Rain",
        "ambient_temperature_c": 32,
        "humidity_percent": 75,
        "package_temperature_c": 5,
        "dispatch_time_utc": "2025-11-12T10:30:00",
        "planned_transit_hours": 28
    }

    response = client.post("/predict", json=shipment)

    assert response.status_code == 422