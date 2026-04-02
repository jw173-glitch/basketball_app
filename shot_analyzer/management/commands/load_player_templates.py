"""
load_player_templates.py
------------------------
Management command that seeds the database with synthetic shooting-form
reference sequences for famous players.

The angle curves are parameterised approximations of each player's
documented shooting mechanics — not extracted from real video footage.
Run once after initial migration:

    python manage.py load_player_templates
"""

import json
import numpy as np
from django.core.management.base import BaseCommand
from shot_analyzer.models import PlayerTemplate


# ── Angle-curve generator ─────────────────────────────────────────────────────

def _smooth_curve(keyframe_values: list, keyframe_positions: list, n: int) -> np.ndarray:
    """Interpolate between keyframe values at given positions (0.0–1.0) over n frames."""
    t = np.linspace(0.0, 1.0, n)
    return np.interp(t, keyframe_positions, keyframe_values)


def _build_sequence(params: dict, fps: float = 30.0, seed: int = 0) -> str:
    """Generate a realistic shooting-motion angle sequence and return it as JSON.

    params keys (all angles in degrees):
        n_frames            total frames
        knee_start          knee angle at address
        knee_min            knee angle at deepest dip (loading)
        knee_end            knee angle at follow-through
        elbow_start         right elbow at address
        elbow_set           right elbow at set point (peak loading)
        elbow_release       right elbow at release
        shoulder_start      right shoulder at address
        shoulder_release    right shoulder at release
        shoulder_follow     right shoulder at follow-through
    """
    rng = np.random.default_rng(seed)
    n = params['n_frames']

    # Phase positions (normalised 0–1):  address · loading · set · rising · release · follow
    ph = [0.0, 0.20, 0.38, 0.55, 0.70, 1.0]

    right_knee = _smooth_curve(
        [params['knee_start'], params['knee_min'],  params['knee_min'],
         params['knee_end'],   params['knee_end'],  params['knee_end']],
        ph, n,
    )
    right_hip = _smooth_curve(
        [params['knee_start'] + 5,  params['knee_min'] + 10, params['knee_min'] + 10,
         params['knee_end'] - 5,    params['knee_end'] - 5,  params['knee_end'] - 5],
        ph, n,
    )
    right_elbow = _smooth_curve(
        [params['elbow_start'], params['elbow_start'], params['elbow_set'],
         params['elbow_set'],   params['elbow_release'], params['elbow_release']],
        ph, n,
    )
    right_shoulder = _smooth_curve(
        [params['shoulder_start'], params['shoulder_start'], params['shoulder_start'] + 15,
         params['shoulder_release'] * 0.6, params['shoulder_release'], params['shoulder_follow']],
        ph, n,
    )
    # Guide arm mirrors shooting arm with reduced range
    left_elbow     = right_elbow     * 0.85 + rng.normal(0, 1, n)
    left_shoulder  = right_shoulder  * 0.80 + rng.normal(0, 1, n)
    left_knee      = right_knee      * 0.98 + rng.normal(0, 0.5, n)
    left_hip       = right_hip       * 0.98 + rng.normal(0, 0.5, n)

    # Add small noise to all curves for realism
    noise_scale = 1.5
    curves = {
        'right_elbow':    right_elbow    + rng.normal(0, noise_scale, n),
        'left_elbow':     left_elbow,
        'right_shoulder': right_shoulder + rng.normal(0, noise_scale, n),
        'left_shoulder':  left_shoulder,
        'right_knee':     right_knee     + rng.normal(0, noise_scale * 0.5, n),
        'left_knee':      left_knee,
        'right_hip':      right_hip      + rng.normal(0, noise_scale * 0.5, n),
        'left_hip':       left_hip,
    }

    frames = [
        {
            'frame_idx': i,
            'angles': {joint: round(float(curves[joint][i]), 2) for joint in curves},
        }
        for i in range(n)
    ]
    return json.dumps({'fps': fps, 'frames': frames})


# ── Player definitions ────────────────────────────────────────────────────────

PLAYERS = [
    {
        'name': 'Stephen Curry',
        'team': 'Golden State Warriors',
        'style_notes': (
            'Quick-release off the catch or dribble. Minimal knee dip keeps '
            'the motion compact and fast. Very high follow-through with the '
            'elbow extending well past the vertical.'
        ),
        'params': dict(
            n_frames=38, knee_start=154, knee_min=108, knee_end=168,
            elbow_start=150, elbow_set=86, elbow_release=170,
            shoulder_start=28, shoulder_release=138, shoulder_follow=130,
        ),
        'seed': 1,
    },
    {
        'name': 'LeBron James',
        'team': 'Los Angeles Lakers',
        'style_notes': (
            'Power-based form with a deep leg drive. More deliberate pace '
            'than Curry. Generates significant upward force from the lower '
            'body before the arm arc begins.'
        ),
        'params': dict(
            n_frames=44, knee_start=158, knee_min=92, knee_end=172,
            elbow_start=155, elbow_set=92, elbow_release=162,
            shoulder_start=22, shoulder_release=128, shoulder_follow=122,
        ),
        'seed': 2,
    },
    {
        'name': 'Kevin Durant',
        'team': 'Phoenix Suns',
        'style_notes': (
            'Extremely high release point due to exceptional height and arm '
            'length. Elbow nearly fully extends at release. Fluid, straight-up '
            'motion with minimal lateral movement.'
        ),
        'params': dict(
            n_frames=46, knee_start=152, knee_min=105, knee_end=166,
            elbow_start=148, elbow_set=80, elbow_release=175,
            shoulder_start=30, shoulder_release=148, shoulder_follow=140,
        ),
        'seed': 3,
    },
    {
        'name': 'Kobe Bryant',
        'team': 'Los Angeles Lakers',
        'style_notes': (
            'Balanced mid-range form adaptable to fadeaways and turnarounds. '
            'Controlled elbow tuck, consistent set point, and reliable '
            'follow-through independent of body lean.'
        ),
        'params': dict(
            n_frames=43, knee_start=153, knee_min=98, knee_end=167,
            elbow_start=152, elbow_set=87, elbow_release=164,
            shoulder_start=27, shoulder_release=133, shoulder_follow=126,
        ),
        'seed': 4,
    },
    {
        'name': 'Classic Form',
        'team': 'Textbook',
        'style_notes': (
            'BEEF technique: Balance, Eyes, Elbow, Follow-through. '
            'Ideal for beginners — neutral stance, elbow under the ball, '
            'full arm extension, wrist flick at release.'
        ),
        'params': dict(
            n_frames=45, knee_start=155, knee_min=100, knee_end=170,
            elbow_start=155, elbow_set=88, elbow_release=166,
            shoulder_start=25, shoulder_release=132, shoulder_follow=125,
        ),
        'seed': 5,
    },
]


# ── Command ───────────────────────────────────────────────────────────────────

class Command(BaseCommand):
    help = 'Seed the database with synthetic player shooting-form templates.'

    def handle(self, *args, **options):
        created = 0
        skipped = 0

        for player in PLAYERS:
            if PlayerTemplate.objects.filter(name=player['name']).exists():
                self.stdout.write(f'  skip  {player["name"]} (already exists)')
                skipped += 1
                continue

            sequence_json = _build_sequence(player['params'], seed=player['seed'])
            PlayerTemplate.objects.create(
                name=player['name'],
                team=player['team'],
                style_notes=player['style_notes'],
                sequence_json=sequence_json,
            )
            self.stdout.write(self.style.SUCCESS(f'  created  {player["name"]}'))
            created += 1

        self.stdout.write(self.style.SUCCESS(
            f'\nDone — {created} created, {skipped} skipped.'
        ))
