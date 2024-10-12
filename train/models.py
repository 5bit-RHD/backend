from django.db import models

# Модель команды, включает все необходимы поля. Нужна для сохранения историй запросов.
class Command(models.Model):
    file = models.FileField(upload_to="uploads", blank=True, null=True)
    command = models.TextField(blank=True)
    label = models.IntegerField(blank=True, null=True)
    attribute = models.TextField(blank=True, null=True)
    time = models.TextField(blank=True, null=True)
    memory = models.TextField(blank=True, null=True)
