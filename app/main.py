from fastapi import FastAPI
from app.core.config import APP_NAME, APP_VERSION
from app.core.event_handlers import startup_handler, shutdown_handler
from app.api.routes import train, prediction, validation


app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.include_router(train.router)
app.include_router(validation.router)
app.include_router(prediction.router)

app.add_event_handler("startup", startup_handler(app))
app.add_event_handler("shutdown", shutdown_handler(app))
