"""
Churn Prediction API

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

from webbrowser import get

from litestar import Litestar
from pydantic import BaseModel

from app.logger_setup import setup_logging

logger = setup_logging()


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    # TODO 1: Add one field (type float) per feature your model expects
    feature1: float
    feature2: float
    pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# TODO 2: Create a GET endpoint at "/" that returns a welcome message
#         Log that the home endpoint was accessed

@get("/")
async def hello_world() -> dict[str, str]:
    return {"message": "Hello, world!"}

app = Litestar(route_handlers=[hello_world])
    

# TODO 3: Create a GET endpoint at "/health" that returns {"status": "healthy"}

@get("/")
async def hello_world() -> dict[str, str]:
    return {"message": "Hello, world!"}

app = Litestar(route_handlers=[hello_world])
    

# TODO 3: Create a GET endpoint at "/health" that returns {"status": "healthy"}

@get("/health")
async def health_check() :
    return {"status": "healthy"}

# TODO 4: Create a POST endpoint at "/predict" that:
#         - Accepts a ChurnRequest as the data parameter
#         - Extracts features into a list
#         - Calls predict_churn(features)
#         - Returns the prediction
#         - Logs the input features and the prediction result


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
# TODO 5: Register your endpoint functions in the list below
app = Litestar(
    route_handlers=[],
)
