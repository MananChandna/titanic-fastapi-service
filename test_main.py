# File: test_main.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint_success():
    # Test with a sample passenger who should not survive
    response = client.post("/predict", json={
        "Pclass": 3,
        "Sex": "male",
        "Age": 35,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 8.05
    })
    # Assert that the API returns a 200 OK status code
    assert response.status_code == 200
    # Assert that the prediction is what we expect
    assert response.json()["prediction"] == "Did not survive"

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running."}
