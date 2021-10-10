import logging

from app.core.config import DATA_MODEL_PATH
from app.services.estimator import Estimator


def load_model(app):
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
    
    
def unload_model(app):
    """Called on application shutdown to clean the context (close db connections, ...)

    Args:
        app (FastAPI): FastAPI application
    """
    
    logging.info("Shuting down the app")
    app.state.model = None


def startup_handler(app):
    def startup():
        load_model(app)

    return startup


def shutdown_handler(app):
    def shutdown():
        unload_model(app)

    return shutdown
