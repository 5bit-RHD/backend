from django.urls import path

from train.views import CommandViewSet

# Импортируем url-ы
urlpatterns = [
    path('command/', CommandViewSet.as_view()),
]
