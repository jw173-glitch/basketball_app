"""
services.py
-----------
Background training worker. Runs in a daemon thread so it does not block
the Django request/response cycle. Communicates results back to the caller
by writing to the PersonalModel row in the database.
"""

import json
import pickle
import threading

import numpy as np
from django.db import close_old_connections
from django.utils import timezone

from core.shot_trainer import ShotModelFitter, InsufficientDataError


def start_training_thread(user_id: int) -> threading.Thread:
    """Spawn a daemon thread that trains the user's personal model.

    The thread updates PersonalModel.training_status as it progresses.
    Returns the thread object (already started)."""
    thread = threading.Thread(
        target=_training_worker,
        args=(user_id,),
        daemon=True,
        name=f'training-user-{user_id}',
    )
    thread.start()
    return thread


def _training_worker(user_id: int) -> None:
    """Background worker: load samples from DB, train model, save results."""
    close_old_connections()

    from .models import PersonalModel, TrainingSample

    def update_status(status: str, message: str, **kwargs):
        PersonalModel.objects.filter(user_id=user_id).update(
            training_status=status,
            training_message=message,
            **kwargs,
        )

    try:
        update_status('running', 'Loading training samples...')

        samples = list(TrainingSample.objects.filter(user_id=user_id))
        if not samples:
            raise InsufficientDataError("No training samples found for this user.")

        X = np.array([json.loads(s.features_json) for s in samples], dtype=np.float32)
        y = np.array([s.label for s in samples], dtype=np.int32)

        update_status('running', f'Fitting model on {len(X)} samples...')

        fitter = ShotModelFitter()
        result = fitter.fit(X, y)

        model_bytes = pickle.dumps({'model': result.model, 'scaler': result.scaler})

        update_status(
            'complete',
            f'Done. Cross-validation accuracy: {result.cv_accuracy:.0%}',
            model_data=model_bytes,
            cv_accuracy=result.cv_accuracy,
            model_type=result.model_type,
            trained_at=timezone.now(),
            n_samples=len(X),
            n_made=int((y == 1).sum()),
            n_missed=int((y == 0).sum()),
        )

    except Exception as exc:
        update_status('error', str(exc))
    finally:
        close_old_connections()
