from fastapi import FastAPI
from .core.config import APP_NAME, APP_VERSION
from .core.event_handlers import startup_handler, shutdown_handler
from .api.routes import train, predict


app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.include_router(train.router)
app.include_router(predict.router)

app.add_event_handler("startup", startup_handler(app))
app.add_event_handler("shutdown", shutdown_handler(app))
