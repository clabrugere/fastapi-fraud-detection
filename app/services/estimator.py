import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.base import clone


class Estimator:
    
    def __init__(self, path=None, *model_args):
        self.path = path
        if self.path is not None:
            self.model = joblib.load(self.path)
        else:
            self.model = LGBMClassifier(*model_args)
    
    def _score(self, y_true, y_hat, reduce=True):
        pass
    
    def fit(self, X, y, *fit_args):
        self.model.fit(X, y,*fit_args)
        
    def save(self, path):
        joblib.dump(self.model, path)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def validate(self, X, y, n_splits: int = 5):
        
        scores = np.zeros(n_splits)
        model = clone(self.model)
        cv = KFold(n_splits=n_splits, shuffle=True)
        
        for i, (idx_train, idx_test) in enumerate(cv.split(X, y)):
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]
            
            model.fit(X_train, y_train)
            y_probs = model.predict_proba(X_test)
            score = self._score(y_test, y_probs)
            scores[i] = score
        
        return scores, n_splits, "metric", self.path