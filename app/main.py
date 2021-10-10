from fastapi import FastAPI

from app.api.routes import prediction, train, validation
from app.core.config import APP_NAME, APP_VERSION
from app.core.events import shutdown_handler, startup_handler

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.include_router(train.router)
app.include_router(validation.router)
app.include_router(prediction.router)

app.add_event_handler("startup", startup_handler(app))
app.add_event_handler("shutdown", shutdown_handler(app))
