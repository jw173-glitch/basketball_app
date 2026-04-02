import base64
import json
import os
import sys
import tempfile

import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consistency_scorer import JOINT_WEIGHTS, ConsistencyScorer
from core.pose_extractor import PoseExtractor, ShotSequence
from core.shot_featurizer import featurize
from core.shot_trainer import ShotModelFitter
from utils.visualizer import (
    JOINT_LABELS,
    fig_to_bytes,
    plot_angle_curves,
    plot_joint_bars,
    plot_radar,
    plot_score_card,
)

from .forms import ScoreShotForm, TrainingSampleForm
from .models import AnalysisSession, PersonalModel, TrainingSample
from .services import start_training_thread


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib Figure to a base64 PNG string, then close the figure."""
    png_bytes = fig_to_bytes(fig)
    plt.close(fig)
    return base64.b64encode(png_bytes).decode('utf-8')


def _save_temp_video(uploaded_file) -> str:
    """Save an uploaded file to a temp path and return that path."""
    suffix = os.path.splitext(uploaded_file.name)[-1] or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in uploaded_file.chunks():
        tmp.write(chunk)
    tmp.flush()
    tmp.close()
    return tmp.name


def _trim_sequence(seq: ShotSequence, start_pct: float, end_pct: float) -> ShotSequence:
    """Return a sub-sequence based on percentage start/end of total frames."""
    n = len(seq.frames)
    start_idx = int(start_pct / 100.0 * n)
    end_idx = max(start_idx + 1, int(end_pct / 100.0 * n))
    trimmed = ShotSequence(fps=seq.fps)
    trimmed.frames = seq.frames[start_idx:end_idx]
    return trimmed


def _get_or_create_personal_model(user) -> PersonalModel:
    model_obj, _ = PersonalModel.objects.get_or_create(user=user)
    return model_obj


# ── Auth ──────────────────────────────────────────────────────────────────────

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'shot_analyzer/register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect(request.GET.get('next', 'home'))
    else:
        form = AuthenticationForm()
    return render(request, 'shot_analyzer/login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')


# ── Home ──────────────────────────────────────────────────────────────────────

@login_required
def home(request):
    recent_sessions = AnalysisSession.objects.filter(user=request.user)[:5]
    personal_model = _get_or_create_personal_model(request.user)
    sample_counts = {
        'total':  TrainingSample.objects.filter(user=request.user).count(),
        'made':   TrainingSample.objects.filter(user=request.user, label=1).count(),
        'missed': TrainingSample.objects.filter(user=request.user, label=0).count(),
    }
    return render(request, 'shot_analyzer/home.html', {
        'recent_sessions': recent_sessions,
        'personal_model': personal_model,
        'sample_counts': sample_counts,
    })


# ── Compare — step 1: extract preview frames ─────────────────────────────────

@login_required
def extract_preview_frames(request):
    """AJAX endpoint: save both uploaded videos to session temp files and return
    evenly-spaced thumbnail frames so the frontend can show a live trim preview."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    if 'ref_video' not in request.FILES or 'user_video' not in request.FILES:
        return JsonResponse({'error': 'Both videos are required.'}, status=400)

    # Remove previous temp files if the user re-uploads
    for key in ('ref_video_path', 'user_video_path'):
        old_path = request.session.get(key)
        if old_path and os.path.exists(old_path):
            try:
                os.unlink(old_path)
            except Exception:
                pass

    ref_path  = _save_temp_video(request.FILES['ref_video'])
    user_path = _save_temp_video(request.FILES['user_video'])

    request.session['ref_video_path']  = ref_path
    request.session['user_video_path'] = user_path
    request.session['ref_video_name']  = request.FILES['ref_video'].name
    request.session['user_video_name'] = request.FILES['user_video'].name

    def _extract_thumbnails(video_path: str, n: int = 21):
        """Return n base64 JPEG thumbnails sampled evenly across the video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        thumbnails = []
        for i in range(n):
            frame_idx = int(i / (n - 1) * (total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                thumb = cv2.resize(frame, (320, int(h * 320 / w)))
                _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
                thumbnails.append(base64.b64encode(buf.tobytes()).decode('utf-8'))
            else:
                thumbnails.append(None)
        cap.release()
        return thumbnails, total_frames

    try:
        ref_thumbs,  ref_total  = _extract_thumbnails(ref_path)
        user_thumbs, user_total = _extract_thumbnails(user_path)
        return JsonResponse({
            'ref_frames':  ref_thumbs,
            'ref_total':   ref_total,
            'user_frames': user_thumbs,
            'user_total':  user_total,
        })
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)


# ── Compare — step 2: run full analysis ───────────────────────────────────────

@login_required
def compare(request):
    if request.method == 'POST':
        ref_path  = request.session.get('ref_video_path')
        user_path = request.session.get('user_video_path')

        if not ref_path or not user_path or \
                not os.path.exists(ref_path) or not os.path.exists(user_path):
            messages.error(request, 'Video files not found. Please upload again.')
            return redirect('compare')

        ref_start_pct  = float(request.POST.get('ref_start_pct',  0))
        ref_end_pct    = float(request.POST.get('ref_end_pct',   100))
        user_start_pct = float(request.POST.get('user_start_pct', 0))
        user_end_pct   = float(request.POST.get('user_end_pct',  100))
        dtw_scale      = int(request.POST.get('dtw_scale', 200))

        try:
            extractor = PoseExtractor()
            ref_seq  = extractor.process_video(ref_path,  annotate=True, max_frames=300)
            user_seq = extractor.process_video(user_path, annotate=True, max_frames=300)

            if len(ref_seq.frames) < 5 or len(user_seq.frames) < 5:
                messages.error(request, 'Not enough pose frames detected. Make sure your full body is visible.')
                return redirect('compare')

            ref_seq  = _trim_sequence(ref_seq,  ref_start_pct,  ref_end_pct)
            user_seq = _trim_sequence(user_seq, user_start_pct, user_end_pct)

            if len(ref_seq.frames) < 5 or len(user_seq.frames) < 5:
                messages.error(request, 'Trimmed selection too short. Widen the range.')
                return redirect('compare')

            raw_weights = {}
            for joint in JOINT_LABELS:
                raw_val = request.POST.get(f'weight_{joint}')
                if raw_val is not None:
                    val = int(raw_val)
                    if val > 0:
                        raw_weights[joint] = val

            if raw_weights:
                total = sum(raw_weights.values())
                joint_weights = {j: v / total for j, v in raw_weights.items()}
            else:
                joint_weights = None

            scorer = ConsistencyScorer(joint_weights=joint_weights, dtw_scale=dtw_scale)
            report = scorer.compare(ref_seq, user_seq)

            joint_scores_data = [
                {
                    'joint':                js.joint,
                    'label':                JOINT_LABELS.get(js.joint, js.joint),
                    'score':                round(js.score, 1),
                    'dtw_distance':         round(js.dtw_distance, 1),
                    'weight':               round(js.weight * 100, 1),
                    'is_most_inconsistent': js.is_most_inconsistent,
                }
                for js in report.joint_scores if js.weight > 0
            ]

            analysis_session = AnalysisSession.objects.create(
                user=request.user,
                ref_video_name=request.session.get('ref_video_name', 'reference'),
                user_video_name=request.session.get('user_video_name', 'your shot'),
                overall_score=report.overall_score,
                grade=report.grade,
                most_inconsistent_joint=report.most_inconsistent_joint,
                most_inconsistent_phase=report.most_inconsistent_phase,
                feedback_json=json.dumps(report.feedback),
                joint_scores_json=json.dumps(joint_scores_data),
                chart_score_card=_fig_to_base64(plot_score_card(report)),
                chart_radar=_fig_to_base64(plot_radar(report)),
                chart_joint_bars=_fig_to_base64(plot_joint_bars(report)),
                chart_angle_curves=_fig_to_base64(plot_angle_curves(ref_seq, user_seq)),
            )
            return redirect('results', session_id=analysis_session.id)

        finally:
            for path in (ref_path, user_path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
            for key in ('ref_video_path', 'user_video_path', 'ref_video_name', 'user_video_name'):
                request.session.pop(key, None)

    return render(request, 'shot_analyzer/compare.html', {'joint_labels': JOINT_LABELS})


# ── Results ───────────────────────────────────────────────────────────────────

@login_required
def results(request, session_id):
    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)
    return render(request, 'shot_analyzer/results.html', {
        'session': session,
        'feedback': json.loads(session.feedback_json),
        'joint_scores': json.loads(session.joint_scores_json),
        'joint_labels': JOINT_LABELS,
    })


# ── History ───────────────────────────────────────────────────────────────────

@login_required
def history(request):
    sessions = AnalysisSession.objects.filter(user=request.user)
    return render(request, 'shot_analyzer/history.html', {'sessions': sessions})


# ── Training ──────────────────────────────────────────────────────────────────

@login_required
def training(request):
    personal_model = _get_or_create_personal_model(request.user)

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'add_sample':
            form = TrainingSampleForm(request.POST, request.FILES)
            if form.is_valid():
                video_path = _save_temp_video(request.FILES['video'])
                try:
                    extractor = PoseExtractor()
                    seq = extractor.process_video(video_path, annotate=False, max_frames=300)
                    if len(seq.frames) < 5:
                        messages.error(request, 'Not enough pose frames detected. Make sure your full body is visible.')
                    else:
                        features = featurize(seq)
                        TrainingSample.objects.create(
                            user=request.user,
                            video_name=request.FILES['video'].name,
                            label=int(form.cleaned_data['label']),
                            features_json=json.dumps(features.tolist()),
                        )
                        n_made   = TrainingSample.objects.filter(user=request.user, label=1).count()
                        n_missed = TrainingSample.objects.filter(user=request.user, label=0).count()
                        personal_model.n_samples = n_made + n_missed
                        personal_model.n_made    = n_made
                        personal_model.n_missed  = n_missed
                        personal_model.save(update_fields=['n_samples', 'n_made', 'n_missed'])
                        messages.success(request, f'Sample added! Total: {n_made + n_missed} ({n_made} made, {n_missed} missed)')
                finally:
                    try:
                        os.unlink(video_path)
                    except Exception:
                        pass
            else:
                messages.error(request, 'Invalid form submission.')

        elif action == 'start_training':
            if personal_model.training_status != 'running':
                personal_model.training_status = 'running'
                personal_model.training_message = 'Queued...'
                personal_model.save(update_fields=['training_status', 'training_message'])
                start_training_thread(request.user.id)

        elif action == 'clear_samples':
            TrainingSample.objects.filter(user=request.user).delete()
            personal_model.n_samples = 0
            personal_model.n_made    = 0
            personal_model.n_missed  = 0
            personal_model.save(update_fields=['n_samples', 'n_made', 'n_missed'])
            messages.success(request, 'All training samples cleared.')

        return redirect('training')

    samples = TrainingSample.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'shot_analyzer/training.html', {
        'personal_model': personal_model,
        'samples': samples,
        'form': TrainingSampleForm(),
        'can_train': personal_model.n_made >= 3 and personal_model.n_missed >= 3,
        'joint_labels': JOINT_LABELS,
    })


@login_required
def training_status_api(request):
    """AJAX endpoint: returns current training status as JSON."""
    personal_model = _get_or_create_personal_model(request.user)
    return JsonResponse({
        'status':     personal_model.training_status,
        'message':    personal_model.training_message,
        'accuracy':   personal_model.accuracy_pct,
        'trained_at': personal_model.trained_at.isoformat() if personal_model.trained_at else None,
    })


# ── Score ─────────────────────────────────────────────────────────────────────

@login_required
def score(request):
    personal_model = _get_or_create_personal_model(request.user)

    if not personal_model.is_trained:
        return render(request, 'shot_analyzer/score.html', {
            'not_trained': True,
            'personal_model': personal_model,
        })

    if request.method == 'POST':
        form = ScoreShotForm(request.POST, request.FILES)
        if form.is_valid():
            video_path = _save_temp_video(request.FILES['video'])
            try:
                extractor = PoseExtractor()
                seq = extractor.process_video(video_path, annotate=False, max_frames=300)
                if len(seq.frames) < 5:
                    messages.error(request, 'Not enough pose frames detected.')
                    return render(request, 'shot_analyzer/score.html', {
                        'form': form, 'personal_model': personal_model,
                    })

                features = featurize(seq)
                fitter = ShotModelFitter()
                prediction = fitter.predict(bytes(personal_model.model_data), features)

                return render(request, 'shot_analyzer/score.html', {
                    'form': form,
                    'personal_model': personal_model,
                    'prediction': prediction,
                    'video_name': request.FILES['video'].name,
                })
            finally:
                try:
                    os.unlink(video_path)
                except Exception:
                    pass
    else:
        form = ScoreShotForm()

    return render(request, 'shot_analyzer/score.html', {
        'form': form,
        'personal_model': personal_model,
    })
