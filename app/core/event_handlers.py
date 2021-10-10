import logging

from app.core.config import DATA_MODEL_PATH
from app.services.estimator import Estimator


def startup_handler(app):
    """Called on application start to load a local pickled model in memory, if it exists

    Args:
        app (FastAPI): FastAPI application
    """
    
    logging.info("Starting up the application")
    model_path = DATA_MODEL_PATH
    
    if model_path.exists():
        model = Estimator(model_path)
        app.state.model = model
        logging.info(f"Loaded model {model_path}")
    else:
        app.state.model = Estimator()
        logging.warning(f"No existing model found in {model_path}")
    
    
def shutdown_handler(app):
    """Called on application shutdown to clean the context (close db connections, ...)

    Args:
        app (FastAPI): FastAPI application
    """
    
    logging.info("Shuting down the app")
    app.state.model = None
