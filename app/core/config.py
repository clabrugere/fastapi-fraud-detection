from pathlib import Path


APP_VERSION = "0.1"
APP_NAME = "fraud-detection"

DATA_X_TRAIN_PATH = Path("https://raw.githubusercontent.com/clabrugere/fastapi-fraud-detection/main/data/X_train.csv?token=AOMGQVORI3XHC7SWFFMGMJDBMLTZ2")
DATA_y_TRAIN_PATH = Path("https://raw.githubusercontent.com/clabrugere/fastapi-fraud-detection/main/data/y_train.csv?token=AOMGQVNKH2RJLDF6CVGHLSLBMLT3E")
DATA_X_TEST_PATH = Path("")
DATA_y_TEST_PATH = Path("")

DATA_MODEL_PATH = Path("../static/model.pkl")