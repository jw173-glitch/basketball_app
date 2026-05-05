"""
shot_trainer.py
---------------
Two fitters:

  ShotModelFitter      — legacy binary classifier (made vs missed).
                         Kept for backward compatibility.

  AdaptiveModelFitter  — one-class SVM trained on made shots only.
                         Stage-aware: uses augmentation for small datasets
                         and switches strategy based on n_made.

                         Stage 1 (1–9)   DTW only — no ML model trained here;
                                         scoring lives in services/views.
                         Stage 2 (10–19) OneClassSVM on augmented data (~60 samples).
                         Stage 3 (20+)   OneClassSVM on real data, no augmentation.

No Django dependencies — all DB interactions live in services.py.
"""

import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC


class InsufficientDataError(ValueError):
    """Raised when there are not enough labeled samples to train."""


class ModelNotTrainedError(RuntimeError):
    """Raised when predict() is called before a model has been trained."""


class ModelVersionError(RuntimeError):
    """Raised when the feature vector dimension does not match the trained model."""


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

    @staticmethod
    def bundle(result: "FitResult", feature_dim: int) -> bytes:
        """Serialize a FitResult to bytes, recording the feature dimension."""
        return pickle.dumps({
            'model':       result.model,
            'scaler':      result.scaler,
            'feature_dim': feature_dim,
        })

    def predict(self, model_bytes: bytes, features: np.ndarray) -> dict:
        """Score a new shot using a previously serialized model bundle.

        Returns a dict with keys: score (0-100), made_prob (0-100), missed_prob (0-100).
        Raises ModelVersionError if the feature dimension does not match the stored model."""
        bundle = pickle.loads(model_bytes)
        clf = bundle['model']
        scaler = bundle['scaler']
        expected_dim = bundle.get('feature_dim')

        if expected_dim is not None and features.shape[0] != expected_dim:
            raise ModelVersionError(
                f"Model was trained on {expected_dim}-dim features; "
                f"got {features.shape[0]}-dim. Please retrain your model."
            )

        X_scaled = scaler.transform(features.reshape(1, -1))
        proba = clf.predict_proba(X_scaled)[0]

        classes = list(clf.classes_)
        made_prob = float(proba[classes.index(1)]) if 1 in classes else 0.0

        return {
            'score': round(made_prob * 100, 1),
            'made_prob': round(made_prob * 100, 1),
            'missed_prob': round((1.0 - made_prob) * 100, 1),
        }


# ── Adaptive one-class fitter ─────────────────────────────────────────────────

@dataclass
class AdaptiveFitResult:
    model: object          # fitted OneClassSVM
    scaler: StandardScaler
    train_score_mean: float
    train_score_std: float
    stage: int
    n_real: int            # real (non-augmented) samples used
    n_total: int           # real + augmented


class AdaptiveModelFitter:
    """
    One-class SVM trained exclusively on made shots.

    Scoring formula (sigmoid shifted so z = 0 → 75):
        z     = (decision_score − real_train_mean) / real_train_std
        score = 100 / (1 + exp(−(z + 1.1)))

    Reference points:
        z = +2  →  95   (well above your average made shot)
        z =  0  →  75   (your typical made shot)
        z = -1  →  52
        z = -2  →  29   (clear outlier)

    mean/std are computed on real shots only so that z = 0 always
    corresponds to the user's actual average, not the synthetic data spread.
    """

    STAGE2_MIN: int = 10   # switch from DTW to OneClassSVM
    STAGE3_MIN: int = 20   # stop augmentation

    @staticmethod
    def stage_for(n_made: int) -> int:
        if n_made >= AdaptiveModelFitter.STAGE3_MIN:
            return 3
        if n_made >= AdaptiveModelFitter.STAGE2_MIN:
            return 2
        return 1

    def fit(
        self,
        X_real: np.ndarray,
        X_augmented: np.ndarray = None,
    ) -> AdaptiveFitResult:
        """
        Train on made shots.

        X_real      : feature matrix for the actual recorded made shots.
        X_augmented : optional synthetic rows (jitter/time-warp).
                      The model trains on real + augmented combined, but the
                      score calibration (mean / std) is computed on X_real only
                      so that z = 0 truly means "your average real made shot".

        Raises InsufficientDataError if len(X_real) < STAGE2_MIN.
        """
        n_real = len(X_real)
        stage  = self.stage_for(n_real)

        if stage < 2:
            raise InsufficientDataError(
                f"Need at least {self.STAGE2_MIN} made shots for ML training, got {n_real}."
            )

        X_train = (
            np.vstack([X_real, X_augmented])
            if X_augmented is not None and len(X_augmented) > 0
            else X_real
        )
        n_total = len(X_train)

        scaler  = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # nu = expected fraction of outliers.
        # Stage 2 uses augmented data with added noise → looser boundary.
        nu    = 0.15 if stage == 2 else 0.10
        model = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
        model.fit(X_train_scaled)

        # Calibrate on REAL shots only so the distribution reflects
        # the user's actual made shots, not the synthetic spread.
        X_real_scaled = scaler.transform(X_real)
        real_scores   = model.decision_function(X_real_scaled)
        t_mean = float(np.mean(real_scores))
        t_std  = float(np.std(real_scores) + 1e-8)

        return AdaptiveFitResult(
            model=model,
            scaler=scaler,
            train_score_mean=t_mean,
            train_score_std=t_std,
            stage=stage,
            n_real=n_real,
            n_total=n_total,
        )

    @staticmethod
    def bundle(result: AdaptiveFitResult, feature_dim: int) -> bytes:
        return pickle.dumps({
            'kind':              'adaptive',
            'model':             result.model,
            'scaler':            result.scaler,
            'feature_dim':       feature_dim,
            'train_score_mean':  result.train_score_mean,
            'train_score_std':   result.train_score_std,
            'stage':             result.stage,
            'n_real':            result.n_real,
            'n_total':           result.n_total,
        })

    def predict(self, model_bytes: bytes, features: np.ndarray) -> dict:
        """
        Score a new shot.  Returns:
            score       0–100   (75 = typical made shot, z=0)
            stage       1/2/3
            method      'one_class_svm'
        Raises ModelVersionError on feature dimension mismatch.
        """
        b = pickle.loads(model_bytes)

        expected_dim = b.get('feature_dim')
        if expected_dim is not None and features.shape[0] != expected_dim:
            raise ModelVersionError(
                f"Model trained on {expected_dim}-dim features; "
                f"got {features.shape[0]}-dim. Please retrain."
            )

        X_scaled = b['scaler'].transform(features.reshape(1, -1))
        raw = float(b['model'].decision_function(X_scaled)[0])

        z     = (raw - b['train_score_mean']) / b['train_score_std']
        score = float(100.0 / (1.0 + np.exp(-(z + 1.1))))

        return {
            'score':  round(score, 1),
            'stage':  b.get('stage', 2),
            'method': 'one_class_svm',
        }
