import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import randint, loguniform, uniform
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


class Estimator:
    
    def __init__(self, path=None):
        self.path = path
        
        if self.path is not None:
            self.model = joblib.load(self.path)
        else:
            self.model = LGBMClassifier(class_weight="balanced", objective="binary")
        
        self.metrics = ["precision", "recall", "roc_auc"]
        self.best_score = None
        self.decision_threshold = .5
    
    def _score(self, y_true, y_proba):
        y_pred = (y_proba >= self.decision_threshold).astype(int)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        return np.array([precision, recall, roc_auc])
    
    def fit(self, X, y, n_splits=3, n_iter=10):
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
            scoring="precision",
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=n_splits),
            refit=True,
            verbose=1
        )
        rand_grid_search.fit(X, y)
        self.best_score = rand_grid_search.best_score_
        self.model = rand_grid_search.best_estimator_

    def save(self, path):
        self.path = path
        joblib.dump(self.model, path)
    
    def predict(self, X):
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= self.decision_threshold).astype(int)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def validate(self, X, y, n_splits=5):
        scores = np.zeros((n_splits, len(self.metrics)))
        model = clone(self.model)
        cv = StratifiedKFold(n_splits=n_splits)

        for i, (idx_train, idx_test) in enumerate(cv.split(X, y)):
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            scores[i, :] = self._score(y_test, y_proba)

        return scores, n_splits, self.metrics, self.path.name
