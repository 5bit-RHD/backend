import os

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from model.predictor import AudioTranscriber
from train.models import Command
from train.serializers import CommandSerializer


class CommandViewSet(APIView):

    audioTranscriber = AudioTranscriber()

    def get(self, request):
        # Получаем и сериализуем все команды по get запросу
        commands = Command.objects.all().order_by('id')
        ser = CommandSerializer(commands, many=True)
        return Response(status=status.HTTP_200_OK, data=ser.data)

    def post(self, request):
        # Валидируем входящие данные
        ser = CommandSerializer(data=request.data)
        if ser.is_valid():
            # Закидываем файл в бд (id присвоится по дефолту)
            command = ser.save()
            # Получаем путь к файлу
            if command.file:
                # Вызываем метод перевода речи в текст, классификации и вычленения атрибутов
                result = self.audioTranscriber.transcribe(command.file.path)
                # Сохраняем результат в базе данных
                command.command = result.get('closest_command', 'unknown')
                print(result.get('text', 'unknown'))
                command.label = result.get('label', 'unknown')
                command.attribute = result.get('attribute', 'unknown')
                command.time = f"{result.get('processing_speed', 0):.2f} сек"
                command.memory = f"{result.get('memory_usage', 0):.2f} МБ"
                command.save()
                # Если всё прошло успешно, вернём 201 статус
                return Response(status=status.HTTP_201_CREATED, data=CommandSerializer(command).data)
            else:
                # Если файла нет, вернём ошибку
                return Response(status=status.HTTP_400_BAD_REQUEST, data={'file': 'Файл обязателен.'})
        return Response(status=status.HTTP_400_BAD_REQUEST, data=ser.errors)

    def put(self, request):
        # Получаем данные из запроса: название команды и её значение
        command_name = request.data.get('command_name')
        command_value = request.data.get('command_value')

        # Проверка, что оба поля присутствуют
        if not command_name or command_value is None:
            return Response(status=status.HTTP_400_BAD_REQUEST,
                            data={'error': 'Необходимо указать команду и её значение.'})

        try:
            # Пытаемся преобразовать значение команды в целое число
            command_value = int(command_value)
        except ValueError:
            return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'Значение команды должно быть числом.'})

        # Добавляем или обновляем команду в словаре через метод pushNewCommand
        self.audioTranscriber.pushNewCommand(command_name, command_value)

        # Возвращаем успешный ответ
        return Response(status=status.HTTP_200_OK,
                        data={'message': f"Команда '{command_name}' успешно добавлена/обновлена."})