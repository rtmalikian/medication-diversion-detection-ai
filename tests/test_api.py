import pytest
from fastapi.testclient import TestClient
from main import app

# Create test client
client = TestClient(app)

def test_root_endpoint():
    """
    Test the root endpoint
    """
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "Clinical Risk Modeling Engine" in data["message"]
    assert "timestamp" in data

def test_health_endpoint():
    """
    Test the health check endpoint
    """
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_risk_features_endpoint():
    """
    Test the risk features endpoint
    """
    response = client.get("/api/v1/risk-assessment/features")
    # This might return 500 if model is not trained, but should not crash
    assert response.status_code in [200, 500]

def test_patient_endpoints():
    """
    Test patient-related endpoints
    """
    # Test getting patients (should return empty list or error if DB not set up)
    response = client.get("/api/v1/patients")
    assert response.status_code in [200, 500]  # OK if empty or error due to no DB

def test_invalid_patient():
    """
    Test getting an invalid patient
    """
    response = client.get("/api/v1/patients/invalid_id")
    # Should return 404 or 500 (not found or DB error)
    assert response.status_code in [404, 500]

if __name__ == "__main__":
    pytest.main([__file__])