import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import randint, loguniform, uniform
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


class FraudDetection(BaseEstimator):

    def __init__(self, path=None) -> None:
        self.path = path

        if self.path is not None:
            self.model = joblib.load(self.path)
        else:
            self.model = LGBMClassifier(class_weight="balanced", objective="binary")

        self.best_score = None

    def __str__(self) -> str:
        if self.path is not None:
            return self.path.name
        else:
            return self.__name__

    def fit(self, X: np.ndarray, y: np.ndarray, finetune: bool = True, n_splits: int = 3, n_iter: int = 10) -> None:
        if finetune:
            lgb_params = {
                'n_estimators': randint(100, 500),
                'num_leaves': randint(2, 100),
                'min_child_samples': randint(50, 500),
                'min_child_weight': loguniform(1e-5, 1e4),
                'subsample': uniform(0.1, 0.9),
                'colsample_bytree': uniform(0.2, 0.8),
                'reg_alpha': uniform(0, 100),
                'reg_lambda': uniform(0, 100)
            }
            rand_grid_search = RandomizedSearchCV(
                self.model,
                lgb_params,
                scoring="recall",
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=n_splits),
                refit=True,
                verbose=1
            )
            rand_grid_search.fit(X, y)
            self.best_score = rand_grid_search.best_score_
            self.model = rand_grid_search.best_estimator_
        else:
            self.model.fit(X, y)

    def save(self, path) -> None:
        self.path = path
        joblib.dump(self.model, path)

    def predict(self, X: np.ndarray, proba_threshold: float = .5) -> np.ndarray:
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= proba_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
