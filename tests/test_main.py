"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""

import logging

import pytest
from litestar.testing import TestClient

from app.logger_setup import setup_logging
from app.model_utils import predict_churn
from main import app

SAMPLE_FEATURES = [619, "France", "Female", 42, 2, 0, 1, 1, 1, 101348.88]
SAMPLE_PAYLOAD = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
}


def test_predict_churn_returns_binary_prediction():
    prediction = predict_churn(SAMPLE_FEATURES)
 
    assert prediction in [0, 1]


def test_predict_churn_raises_for_wrong_feature_count():
    with pytest.raises(ValueError):
        predict_churn([619, "France"])


def test_setup_logging_returns_named_logger():
    logger = setup_logging()

    assert isinstance(logger, logging.Logger)
    assert logger.name == "churn_prediction_api"


def test_predict_endpoint_returns_prediction():
    with TestClient(app=app) as client:
        response = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert response.status_code == 201
    assert response.json()["prediction"] in [0, 1]


def test_health_endpoint_returns_healthy():
    with TestClient(app=app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_home_endpoint_returns_welcome_message():
    with TestClient(app=app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Churn Prediction API"}


def test_predict_endpoint_rejects_invalid_input():
    invalid_payload = SAMPLE_PAYLOAD | {"CreditScore": "not-a-number"}

    with TestClient(app=app) as client:
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 400
