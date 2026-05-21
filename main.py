"""
Churn Prediction API

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

from litestar import Litestar, get, post
from pydantic import BaseModel

from app.logger_setup import setup_logging
from app.model_utils import predict_churn

logger = setup_logging()


class ChurnRequest(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float


@get("/")
async def home() -> dict[str, str]:
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the Churn Prediction API"}


@get("/health")
async def health_check() -> dict[str, str]:
    logger.info("Health endpoint accessed")
    return {"status": "healthy"}


@post("/predict")
async def predict(data: ChurnRequest) -> dict[str, int]:
    features = [
        data.CreditScore,
        data.Geography,
        data.Gender,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary,
    ]

    logger.info("Prediction requested with features: %s", features)
    prediction = predict_churn(features)
    logger.info("Prediction completed with result: %s", prediction)

    return {"prediction": prediction}


app = Litestar(
    route_handlers=[
        home,
        health_check,
        predict,
    ],
    debug=True,
)
