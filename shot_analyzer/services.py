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

from core.shot_trainer import (
    AdaptiveModelFitter, InsufficientDataError,
    ShotModelFitter,  # kept for legacy bundles
)
from core.shot_augmenter import augment_sequence, n_augments_for


def start_training_thread(user_id: int) -> threading.Thread:
    """Spawn a daemon thread that trains the user's personal model."""
    thread = threading.Thread(
        target=_training_worker,
        args=(user_id,),
        daemon=True,
        name=f'training-user-{user_id}',
    )
    thread.start()
    return thread


def auto_train_if_ready(user_id: int, personal_model) -> bool:
    """
    Start a training thread if there is at least 1 made shot and training
    is not already running.  Returns True if training was started.
    """
    if personal_model.training_status == 'running':
        return False
    if personal_model.n_made < 1:
        return False
    personal_model.training_status = 'running'
    personal_model.training_message = 'Auto-training with latest samples…'
    personal_model.save(update_fields=['training_status', 'training_message'])
    start_training_thread(user_id)
    return True


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
    """
    Background worker — adaptive, made-shots-only pipeline.

    Stage 1 (1–9 made):   compute personal reference only; no ML model.
    Stage 2 (10–19 made): augment to ~60 samples, train OneClassSVM.
    Stage 3 (20+ made):   train OneClassSVM on real data, no augmentation.
    """
    close_old_connections()

    from core.shot_featurizer import featurize, FEATURE_DIM
    from .models import PersonalModel, TrainingSample

    def update_status(status: str, message: str, **kwargs):
        PersonalModel.objects.filter(user_id=user_id).update(
            training_status=status,
            training_message=message,
            **kwargs,
        )

    try:
        # ── Load made shots only ──────────────────────────────────────────────
        update_status('running', 'Loading made shots…')
        made_samples = list(
            TrainingSample.objects.filter(user_id=user_id, label=1)
        )
        all_samples = list(TrainingSample.objects.filter(user_id=user_id))

        n_made   = len(made_samples)
        n_missed = len(all_samples) - n_made

        if n_made == 0:
            raise InsufficientDataError("No made shots found. Upload at least one made shot.")

        stage = AdaptiveModelFitter.stage_for(n_made)

        # ── Always: personal reference (average of made shots) ────────────────
        update_status('running', 'Building personal reference sequence…')
        made_sequences = [
            _deserialize_sequence(s.sequence_json)
            for s in made_samples if s.sequence_json
        ]
        ref_json = ''
        if made_sequences:
            reference_seq = compute_personal_reference(made_sequences)
            if reference_seq is not None:
                ref_json = serialize_sequence(reference_seq)

        # ── Stage 1: reference only, no ML ───────────────────────────────────
        if stage == 1:
            update_status(
                'complete',
                f'Stage 1 ({n_made} made shots): scoring via DTW consistency.',
                model_data=None,
                cv_accuracy=None,
                model_type='DTW',
                trained_at=timezone.now(),
                n_samples=n_made,
                n_made=n_made,
                n_missed=n_missed,
                reference_sequence_json=ref_json,
            )
            return

        # ── Stage 2/3: build feature matrix, optionally augment ───────────────
        update_status('running', f'Featurizing {n_made} made shots…')

        raw_features = [json.loads(s.features_json) for s in made_samples]
        dims = {len(f) for f in raw_features}
        if len(dims) > 1:
            raise ValueError(
                f"Made-shot feature dimensions are mixed {dims}. "
                "Please clear samples and re-add them."
            )

        X_real = np.array(raw_features, dtype=np.float32)

        X_aug = None
        if stage == 2:
            n_aug = n_augments_for(n_made, target=60)
            update_status('running',
                          f'Augmenting {n_made} shots × {n_aug} → {n_made*(n_aug+1)} total…')
            aug_seqs = []
            for s in made_sequences:
                aug_seqs.extend(augment_sequence(s, n=n_aug))
            aug_rows = [featurize(s) for s in aug_seqs if len(s.frames) >= 5]
            if aug_rows:
                X_aug = np.array(aug_rows, dtype=np.float32)

        # ── Train OneClassSVM ─────────────────────────────────────────────────
        n_total = n_made + (len(X_aug) if X_aug is not None else 0)
        update_status('running',
                      f'Training OneClassSVM on {n_total} samples (stage {stage})…')
        fitter = AdaptiveModelFitter()
        # X_real and X_aug kept separate so calibration uses real shots only
        result = fitter.fit(X_real, X_augmented=X_aug)
        model_bytes = AdaptiveModelFitter.bundle(result, feature_dim=X_real.shape[1])

        n_synthetic = n_total - n_made
        update_status(
            'complete',
            f'Stage {stage} — {n_made} real'
            + (f' + {n_synthetic} synthetic' if n_synthetic > 0 else '')
            + '. Model ready.',
            model_data=model_bytes,
            cv_accuracy=None,
            model_type=f'OneClassSVM-stage{stage}',
            trained_at=timezone.now(),
            n_samples=n_made,
            n_made=n_made,
            n_missed=n_missed,
            reference_sequence_json=ref_json,
        )

    except Exception as exc:
        update_status('error', str(exc))
    finally:
        close_old_connections()
