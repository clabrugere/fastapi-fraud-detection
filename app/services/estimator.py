import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real


class Estimator:
    
    def __init__(self, path=None, *model_args):
        self.path = path
        
        if self.path is not None:
            self.model = joblib.load(self.path)
        else:
            self.model = LGBMClassifier(class_weight="balanced", objective="binary")
        
        self.metrics = ["precision", "recall", "roc_auc"]
    
    def _score(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        
        return np.array([precision, recall, roc_auc])
    
    def fit(self, X, y, n_splits=3):
        lgb_params = {
            'n_estimators': Integer(100, 1000),
            'num_leaves': Integer(2, 100),
            'min_child_samples': Integer(50, 500),
            'min_child_weight': Real(1e-5, 1e4, prior='log-uniform'),
            'subsample': Real(0.2, 0.8),
            'colsample_bytree': Real(0.4, 0.6),
            'reg_alpha': Real(0, 100, prior='log-uniform'),
            'reg_lambda': Real(0, 100, prior='log-uniform')
        }
        bayesian_opt = BayesSearchCV(
            self.model,
            lgb_params,
            n_iter=50,
            cv=StratifiedKFold(n_splits=n_splits)
        )
        bayesian_opt.fit(X, y)
        self.model = bayesian_opt.best_estimator_
        
    def save(self, path):
        joblib.dump(self.model, path)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y, n_splits: int = 5):
        
        scores = np.zeros((n_splits, len(self.metrics)))
        model = clone(self.model)
        cv = StratifiedKFold(n_splits=n_splits)
        
        for i, (idx_train, idx_test) in enumerate(cv.split(X, y)):
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]
            
            model.fit(X_train, y_train)
            y_probs = model.predict_proba(X_test)
            scores[i, :] = self._score(y_test, y_probs)
        
        return scores, n_splits, self.metrics, self.path
