"""
add_player_template.py
-----------------------
Extract a shooting-form reference template from a real video file and
save it to the PlayerTemplate table.

Usage:
    python manage.py add_player_template \\
        --video /path/to/curry_shot.mp4 \\
        --name "Stephen Curry" \\
        --team "Golden State Warriors" \\
        --notes "Quick release, minimal knee dip, high follow-through." \\
        --start 15 \\   # optional: trim start % (default 0)
        --end   85       # optional: trim end   % (default 100)

    # Overwrite an existing template with the same name:
    python manage.py add_player_template --video ... --name "Stephen Curry" --overwrite
"""

import json
import sys
import os

from django.core.management.base import BaseCommand, CommandError

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.pose_extractor import PoseExtractor, ShotSequence
from shot_analyzer.models import PlayerTemplate


def _trim(seq: ShotSequence, start_pct: float, end_pct: float) -> ShotSequence:
    n = len(seq.frames)
    start_idx = int(start_pct / 100.0 * n)
    end_idx   = max(start_idx + 1, int(end_pct / 100.0 * n))
    trimmed = ShotSequence(fps=seq.fps)
    trimmed.frames = seq.frames[start_idx:end_idx]
    return trimmed


def _serialize(seq: ShotSequence) -> str:
    """Serialize a ShotSequence to JSON (angles only, no image data)."""
    return json.dumps({
        'fps': seq.fps,
        'frames': [
            {'frame_idx': f.frame_idx, 'angles': f.angles}
            for f in seq.frames
        ],
    })


class Command(BaseCommand):
    help = 'Extract pose data from a video and save it as a PlayerTemplate.'

    def add_arguments(self, parser):
        parser.add_argument('--video',     required=True,  help='Path to the shot video file')
        parser.add_argument('--name',      required=True,  help='Player name, e.g. "Stephen Curry"')
        parser.add_argument('--team',      default='',     help='Team name')
        parser.add_argument('--notes',     default='',     help='Short style description')
        parser.add_argument('--start',     type=float, default=0.0,
                            help='Trim start percentage (0-100, default 0)')
        parser.add_argument('--end',       type=float, default=100.0,
                            help='Trim end percentage (0-100, default 100)')
        parser.add_argument('--photo',     default='',
                            help='Path to a player photo (JPG/PNG)')
        parser.add_argument('--overwrite', action='store_true',
                            help='Replace existing template with the same name')

    def handle(self, *args, **options):
        video_path = options['video']
        player_name = options['name']

        if not os.path.exists(video_path):
            raise CommandError(f'Video file not found: {video_path}')

        # Check for existing template
        existing = PlayerTemplate.objects.filter(name=player_name).first()
        if existing and not options['overwrite']:
            raise CommandError(
                f'A template named "{player_name}" already exists. '
                f'Use --overwrite to replace it.'
            )

        self.stdout.write(f'Extracting pose from: {video_path}')
        self.stdout.write('This may take 30-60 seconds...')

        extractor = PoseExtractor(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        seq = extractor.process_video(video_path, annotate=False, max_frames=300)

        if len(seq.frames) < 5:
            raise CommandError(
                f'Only {len(seq.frames)} pose frames detected. '
                f'Make sure the full body is visible in the video.'
            )

        self.stdout.write(f'Detected {len(seq.frames)} frames.')

        # Trim if requested
        start_pct = options['start']
        end_pct   = options['end']
        if start_pct != 0.0 or end_pct != 100.0:
            seq = _trim(seq, start_pct, end_pct)
            self.stdout.write(
                f'Trimmed to {start_pct}%–{end_pct}%  →  {len(seq.frames)} frames kept.'
            )

        if len(seq.frames) < 5:
            raise CommandError('Trimmed sequence is too short (< 5 frames). Widen the range.')

        sequence_json = _serialize(seq)

        # Handle optional photo
        photo_path = options.get('photo', '')
        photo_file = None
        if photo_path:
            if not os.path.exists(photo_path):
                raise CommandError(f'Photo file not found: {photo_path}')
            from django.core.files import File
            photo_file = File(open(photo_path, 'rb'), name=os.path.basename(photo_path))

        if existing:
            existing.team          = options['team']
            existing.style_notes   = options['notes']
            existing.sequence_json = sequence_json
            if photo_file:
                existing.photo.save(photo_file.name, photo_file, save=False)
            existing.save()
            self.stdout.write(self.style.SUCCESS(
                f'Updated template: "{player_name}" ({len(seq.frames)} frames)'
            ))
        else:
            template = PlayerTemplate(
                name=player_name,
                team=options['team'],
                style_notes=options['notes'],
                sequence_json=sequence_json,
            )
            if photo_file:
                template.photo.save(photo_file.name, photo_file, save=False)
            template.save()
            self.stdout.write(self.style.SUCCESS(
                f'Created template: "{player_name}" ({len(seq.frames)} frames)'
            ))
