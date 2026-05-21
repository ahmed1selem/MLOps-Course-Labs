"""
Logging configuration.
"""

import logging


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(name)s - %(message)s",
    )
    return logging.getLogger("churn_prediction_api")
