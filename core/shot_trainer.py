"""
shot_trainer.py
---------------
Pure ML logic: fit a binary classifier (made vs missed) on extracted
shot feature vectors and predict made-probability for new shots.
No Django dependencies — all DB interactions live in services.py.
"""

import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class InsufficientDataError(ValueError):
    """Raised when there are not enough labeled samples to train."""


class ModelNotTrainedError(RuntimeError):
    """Raised when predict() is called before a model has been trained."""


@dataclass
class FitResult:
    model: object
    scaler: StandardScaler
    cv_accuracy: float
    model_type: str   # "SVC" or "LogisticRegression"


class ShotModelFitter:
    """Fit a binary classifier and produce a pickled model bundle."""

    MIN_SAMPLES_TOTAL: int = 6
    MIN_PER_CLASS: int = 3
    SVC_THRESHOLD: int = 15   # use SVC when n_samples >= this

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        """Train on feature matrix X (n_samples x 58) and labels y (0/1).

        Returns a FitResult containing the fitted model, scaler, leave-one-out
        cross-validation accuracy, and the model type string."""
        if len(X) < self.MIN_SAMPLES_TOTAL:
            raise InsufficientDataError(
                f"Need at least {self.MIN_SAMPLES_TOTAL} samples total, got {len(X)}"
            )

        n_made = int((y == 1).sum())
        n_missed = int((y == 0).sum())

        if n_made < self.MIN_PER_CLASS:
            raise InsufficientDataError(
                f"Need at least {self.MIN_PER_CLASS} made shots, got {n_made}"
            )
        if n_missed < self.MIN_PER_CLASS:
            raise InsufficientDataError(
                f"Need at least {self.MIN_PER_CLASS} missed shots, got {n_missed}"
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if len(X) >= self.SVC_THRESHOLD:
            model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            model_type = 'SVC'
        else:
            model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            model_type = 'LogisticRegression'

        loo = LeaveOneOut()
        cv_scores = cross_val_score(model, X_scaled, y, cv=loo, scoring='accuracy')
        cv_accuracy = float(cv_scores.mean())

        model.fit(X_scaled, y)

        return FitResult(
            model=model,
            scaler=scaler,
            cv_accuracy=cv_accuracy,
            model_type=model_type,
        )

    def predict(self, model_bytes: bytes, features: np.ndarray) -> dict:
        """Score a new shot using a previously serialized model bundle.

        Returns a dict with keys: score (0-100), made_prob (0-100), missed_prob (0-100)."""
        bundle = pickle.loads(model_bytes)
        clf = bundle['model']
        scaler = bundle['scaler']

        X_scaled = scaler.transform(features.reshape(1, -1))
        proba = clf.predict_proba(X_scaled)[0]

        classes = list(clf.classes_)
        made_prob = float(proba[classes.index(1)]) if 1 in classes else 0.0

        return {
            'score': round(made_prob * 100, 1),
            'made_prob': round(made_prob * 100, 1),
            'missed_prob': round((1.0 - made_prob) * 100, 1),
        }
