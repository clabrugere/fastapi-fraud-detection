from typing import Tuple, List

import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def validate(
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        proba_threshold: float = .5
) -> Tuple[np.ndarray, int, List[str], str]:

    metrics = ["precision", "recall", "roc_auc"]

    scores = np.zeros((n_splits, len(metrics)))
    model = clone(model)
    cv = StratifiedKFold(n_splits=n_splits)

    for i, (idx_train, idx_test) in enumerate(cv.split(X, y)):
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        model.fit(X_train, y_train, finetune=False)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= proba_threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        scores[i, :] = np.array([precision, recall, roc_auc])

    return scores, n_splits, metrics, str(model)
