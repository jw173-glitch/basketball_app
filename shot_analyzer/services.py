"""
services.py
-----------
Background training worker. Runs in a daemon thread so it does not block
the Django request/response cycle. Communicates results back to the caller
by writing to the PersonalModel row in the database.

After training the ML classifier, the worker also computes a personal
reference sequence by averaging all of the user's made-shot angle series.
This reference is stored in PersonalModel.reference_sequence_json and used
later to score new shots for consistency without requiring a reference video.
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


def serialize_sequence(seq) -> str:
    """Serialize a ShotSequence to a compact JSON string (angles only, no images).
    Compatible with PlayerTemplate.sequence_json format."""
    return json.dumps({
        'fps': seq.fps,
        'frames': [
            {'frame_idx': f.frame_idx, 'angles': f.angles}
            for f in seq.frames
        ],
    })


def _deserialize_sequence(sequence_json: str):
    """Reconstruct a lightweight ShotSequence from stored JSON."""
    from core.pose_extractor import FrameData, ShotSequence
    data = json.loads(sequence_json)
    seq = ShotSequence(fps=data['fps'])
    for f in data['frames']:
        seq.frames.append(FrameData(
            frame_idx=f['frame_idx'],
            landmarks={},
            angles=f['angles'],
            image=None,
        ))
    return seq


def _resample_sequence(seq, target_len: int = 100):
    """Resample a ShotSequence to exactly target_len frames via linear interpolation.
    This makes sequences of different lengths comparable for averaging."""
    from core.pose_extractor import FrameData, ShotSequence

    n = len(seq.frames)
    if n == 0:
        return seq

    joints = list(seq.frames[0].angles.keys())
    old_x = np.linspace(0.0, 1.0, n)
    new_x = np.linspace(0.0, 1.0, target_len)

    resampled_angles: dict = {}
    for joint in joints:
        series = np.array([f.angles.get(joint, np.nan) for f in seq.frames], dtype=float)
        nan_mask = np.isnan(series)
        if nan_mask.all():
            series = np.zeros(n)
        elif nan_mask.any():
            valid_x = old_x[~nan_mask]
            valid_y = series[~nan_mask]
            series[nan_mask] = np.interp(old_x[nan_mask], valid_x, valid_y)
        resampled_angles[joint] = np.interp(new_x, old_x, series)

    result = ShotSequence(fps=seq.fps)
    for i in range(target_len):
        result.frames.append(FrameData(
            frame_idx=i,
            landmarks={},
            angles={j: float(resampled_angles[j][i]) for j in joints},
            image=None,
        ))
    return result


def compute_personal_reference(made_sequences: list, target_len: int = 100):
    """Average multiple made-shot sequences into a single personal reference sequence.

    All sequences are resampled to target_len frames first so they can be
    averaged frame-by-frame. Returns a ShotSequence or None if input is empty."""
    if not made_sequences:
        return None

    resampled = [_resample_sequence(s, target_len) for s in made_sequences]
    joints = list(resampled[0].frames[0].angles.keys())

    avg_angles: dict = {}
    for joint in joints:
        matrix = np.array([
            [f.angles.get(joint, 0.0) for f in s.frames]
            for s in resampled
        ])
        avg_angles[joint] = np.mean(matrix, axis=0)

    from core.pose_extractor import FrameData, ShotSequence
    reference = ShotSequence(fps=30.0)
    for i in range(target_len):
        reference.frames.append(FrameData(
            frame_idx=i,
            landmarks={},
            angles={j: float(avg_angles[j][i]) for j in joints},
            image=None,
        ))
    return reference


def _training_worker(user_id: int) -> None:
    """Background worker: load samples from DB, train ML model, then compute
    and store a personal reference sequence from the user's made shots."""
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

        # Build personal reference from made-shot sequences
        update_status('running', 'Computing personal reference sequence...')
        made_samples = [s for s in samples if s.label == 1 and s.sequence_json]
        ref_json = ''
        if made_samples:
            made_sequences = [_deserialize_sequence(s.sequence_json) for s in made_samples]
            reference_seq = compute_personal_reference(made_sequences)
            if reference_seq is not None:
                ref_json = serialize_sequence(reference_seq)

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
            reference_sequence_json=ref_json,
        )

    except Exception as exc:
        update_status('error', str(exc))
    finally:
        close_old_connections()
