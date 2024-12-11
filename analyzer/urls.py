from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.analyze_sentiment, name='analyze'),
]