from django.urls import path

from . import views

urlpatterns = [
    path('',                    views.home,                name='home'),
    path('register/',           views.register_view,       name='register'),
    path('login/',              views.login_view,          name='login'),
    path('logout/',             views.logout_view,         name='logout'),
    path('compare/',            views.compare,                  name='compare'),
    path('compare/preview/',    views.extract_preview_frames,   name='compare_preview'),
    path('results/<int:session_id>/', views.results,       name='results'),
    path('history/',            views.history,             name='history'),
    path('training/',           views.training,            name='training'),
    path('training/status/',    views.training_status_api, name='training_status'),
    path('score/',              views.score,               name='score'),
    path('detect/',             views.detect_shots,        name='detect_shots'),
]
