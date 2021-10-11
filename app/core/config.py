from pathlib import Path

APP_VERSION = "0.1"
APP_NAME = "fraud-detection"

DATA_X_TRAIN_PATH = r"https://raw.githubusercontent.com/clabrugere/fastapi-fraud-detection/main/data/X_train.csv"
DATA_y_TRAIN_PATH = r"https://raw.githubusercontent.com/clabrugere/fastapi-fraud-detection/main/data/y_train.csv"
DATA_X_VAL_PATH = r"https://raw.githubusercontent.com/clabrugere/fastapi-fraud-detection/main/data/X_val.csv"
DATA_y_VAL_PATH = r"https://raw.githubusercontent.com/clabrugere/fastapi-fraud-detection/main/data/y_val.csv"

DATA_MODEL_PATH = Path(r"app/static/model.pkl")
