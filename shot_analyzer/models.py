from django.db import models
from django.contrib.auth.models import User


class PlayerTemplate(models.Model):
    """Pre-loaded reference shooting sequence for a famous player."""
    name = models.CharField(max_length=100)
    team = models.CharField(max_length=100)
    style_notes = models.TextField(blank=True)
    photo = models.ImageField(upload_to='player_photos/', blank=True, null=True)
    sequence_json = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f'{self.name} ({self.team})'


class AnalysisSession(models.Model):
    """Stores one completed comparison analysis between a reference and user shot."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    created_at = models.DateTimeField(auto_now_add=True)
    ref_video_name = models.CharField(max_length=255)
    user_video_name = models.CharField(max_length=255)
    overall_score = models.FloatField()
    grade = models.CharField(max_length=5)
    most_inconsistent_joint = models.CharField(max_length=50)
    feedback_json = models.TextField()       # JSON list[str]
    joint_scores_json = models.TextField()   # JSON list[dict]
    chart_score_card = models.TextField()    # base64 PNG
    chart_radar = models.TextField()         # base64 PNG
    chart_joint_bars = models.TextField()    # base64 PNG
    chart_angle_curves = models.TextField(blank=True)  # base64 PNG

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username} | {self.overall_score:.1f} | {self.created_at:%Y-%m-%d %H:%M}'


class TrainingSample(models.Model):
    """One labeled shot video (made=1 / missed=0) contributed to a user's personal model."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='training_samples')
    video_name = models.CharField(max_length=255)
    label = models.IntegerField()            # 1 = made, 0 = missed
    features_json = models.TextField()       # JSON list[float], 58-dim vector
    sequence_json = models.TextField(blank=True, default='')  # serialized angle time series
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        outcome = 'Made' if self.label == 1 else 'Missed'
        return f'{self.user.username} | {outcome} | {self.video_name}'


class PersonalModel(models.Model):
    """Per-user trained ML model and its training metadata."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='personal_model')
    model_data = models.BinaryField(null=True, blank=True)   # pickled {model, scaler}
    n_samples = models.IntegerField(default=0)
    n_made = models.IntegerField(default=0)
    n_missed = models.IntegerField(default=0)
    cv_accuracy = models.FloatField(null=True, blank=True)
    model_type = models.CharField(max_length=50, blank=True)
    trained_at = models.DateTimeField(null=True, blank=True)
    training_status = models.CharField(max_length=20, default='idle')
    # idle | running | complete | error
    training_message = models.TextField(blank=True)
    reference_sequence_json = models.TextField(blank=True, default='')  # averaged made-shot sequence

    def __str__(self):
        return f'{self.user.username} | {self.training_status}'

    @property
    def is_trained(self):
        return self.model_data is not None and self.training_status == 'complete'

    @property
    def accuracy_pct(self):
        if self.cv_accuracy is None:
            return None
        return round(self.cv_accuracy * 100, 1)
