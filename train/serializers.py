from rest_framework import serializers

from train.models import Command


class CommandSerializer(serializers.ModelSerializer):
    # Сериализатор модели. Выдаём и требуем все поля для создания и отдачи.
    class Meta:
        model = Command
        fields = "__all__"
