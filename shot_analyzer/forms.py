from django import forms

from utils.visualizer import JOINT_LABELS

DEFAULT_JOINT_WEIGHTS = {
    "right_elbow":    25,
    "right_shoulder": 20,
    "right_knee":     20,
    "right_hip":      15,
    "left_elbow":     10,
    "left_shoulder":   5,
    "left_knee":       5,
    "left_hip":        0,
}


class CompareForm(forms.Form):
    ref_video = forms.FileField(label='Reference Video (pro / standard form)')
    user_video = forms.FileField(label='Your Shot Video')

    dtw_scale = forms.IntegerField(
        min_value=50, max_value=500, initial=200,
        label='DTW Sensitivity (lower = stricter)',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '50'}),
    )

    ref_start_pct = forms.FloatField(
        min_value=0, max_value=100, initial=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '1'}),
        label='Reference: start (%)',
    )
    ref_end_pct = forms.FloatField(
        min_value=0, max_value=100, initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '1'}),
        label='Reference: end (%)',
    )
    user_start_pct = forms.FloatField(
        min_value=0, max_value=100, initial=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '1'}),
        label='Your shot: start (%)',
    )
    user_end_pct = forms.FloatField(
        min_value=0, max_value=100, initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '1'}),
        label='Your shot: end (%)',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for joint, label in JOINT_LABELS.items():
            self.fields[f'weight_{joint}'] = forms.IntegerField(
                min_value=0,
                max_value=100,
                initial=DEFAULT_JOINT_WEIGHTS.get(joint, 10),
                required=False,
                label=label,
                widget=forms.NumberInput(attrs={
                    'class': 'form-control form-control-sm',
                    'type': 'range',
                    'min': '0',
                    'max': '100',
                }),
            )

    def joint_weight_fields(self):
        """Yield (field, joint_name) pairs for rendering joint weight sliders."""
        for joint in JOINT_LABELS:
            yield self[f'weight_{joint}'], joint


class TrainingSampleForm(forms.Form):
    video = forms.FileField(
        label='Shot Video',
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': 'video/*'}),
    )
    label = forms.ChoiceField(
        choices=[('1', 'Made'), ('0', 'Missed')],
        widget=forms.RadioSelect,
        label='Outcome',
    )


class ScoreShotForm(forms.Form):
    video = forms.FileField(
        label='Shot Video',
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': 'video/*'}),
    )
