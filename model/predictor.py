import os
import time
from typing import Dict

import librosa
import numpy as np
import onnxruntime as ort
import psutil
from transformers import Wav2Vec2Processor
from fuzzywuzzy import process, fuzz


class AudioTranscriber:
    """Класс для транскрибации аудиофайлов с использованием модели Wav2Vec2."""

    def __init__(
            self,
            model_name: str = "bond005/wav2vec2-base-ru",
    ) -> None:
        """Инициализирует AudioTranscriber.

        Args:
            model_path: Путь к ONNX модели.
            model_name: Имя модели для загрузки процессора.
        """
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 8
        sess_options.inter_op_num_threads = 8
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path_os = os.path.join(current_dir, "wav2vec2_russian_train_1.onnx")
        self.ort_session = ort.InferenceSession(model_path_os, sess_options)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        self.commands_dict = {'назад с башмака': 17, 'прекратить зарядку тормозной магистрали': 20,
                              'зарядка тормозной магистрали': 6, 'вышел из межвагонного пространства': 7,
                              'вперед на башмак': 15, 'вперед с башмака': 19, 'остановка': 14, 'отцепка': 11,
                              'отказ': 0, 'растянуть автосцепки': 9, 'продолжаем роспуск': 8,
                              'захожу в межвагонное пространство': 13, 'тормозить': 21, 'отмена': 1,
                              'тише': 18, 'отпустить': 22, 'начать осаживание': 3, 'сжать автосцепки': 16,
                              'продолжаем осаживание': 5, 'назад на башмак': 12, 'подтверждение': 2}

        self.count_commands_dict = {'протянуть': 10, 'осадить': 4}

        self.numbers_dict = {
            'один': {"count": 1, "text": "один вагон"}, 'два': {"count": 2, "text": "два вагона"},
            'три': {"count": 3, "text": "три вагона"}, 'четыре': {"count": 4, "text": "четыре вагона"},
            'пять': {"count": 5, "text": "пять вагонов"}, 'шесть': {"count": 6, "text": "шесть вагонов"},
            'семь': {"count": 7, "text": "семь вагонов"}, 'восемь': {"count": 8, "text": "восемь вагонов"},
            'девять': {"count": 9, "text": "девять вагонов"}, 'десять': {"count": 10, "text": "десять вагонов"},
            'одиннадцать': {"count": 11, "text": "одиннадцать вагонов"},
            'двенадцать': {"count": 12, "text": "двенадцать вагонов"},
            'тринадцать': {"count": 13, "text": "тринадцать вагонов"},
            'четырнадцать': {"count": 14, "text": "четырнадцать вагонов"},
            'пятнадцать': {"count": 15, "text": "пятнадцать вагонов"},
            'шестнадцать': {"count": 16, "text": "шестнадцать вагонов"},
            'семнадцать': {"count": 17, "text": "семнадцать вагонов"},
            'восемнадцать': {"count": 18, "text": "восемнадцать вагонов"},
            'девятнадцать': {"count": 19, "text": "девятнадцать вагонов"},
            'двадцать': {"count": 20, "text": "двадцать"},
            'тридцать': {"count": 30, "text": "тридцать"}, 'сорок': {"count": 40, "text": "сорок"},
            'девяносто': {"count": 90, "text": "девяносто"}, 'сто': {"count": 100, "text": "сто"},
            'двести': {"count": 200, "text": "двести"}, 'триста': {"count": 300, "text": "триста"},
            'четыреста': {"count": 400, "text": "четыреста"}, 'пятьсот': {"count": 500, "text": "пятьсот"},
            'шестьсот': {"count": 600, "text": "шестьсот"}, 'семьсот': {"count": 700, "text": "семьсот"},
            'восемьсот': {"count": 800, "text": "восемьсот"}, 'девятьсот': {"count": 900, "text": "девятьсот"},
            'тысяча': {"count": 1000, "text": "тысяча"}
        }

    def pushNewCommand(self, command_name: str, command_value: int) -> None:
        self.commands_dict[command_name] = command_value

    def find_closest_command(self, transcription: str, threshold: int = 90) -> tuple:
        closest_command = process.extractOne(transcription, self.commands_dict.keys(), scorer=fuzz.ratio)
        command = closest_command[0]
        if closest_command[1] >= threshold:
            return command, self.commands_dict[command], -1

        words = transcription.lower().split()
        for word in words:
            closest_word = process.extractOne(word, self.count_commands_dict.keys(), scorer=fuzz.ratio)
            if closest_word[1] >= 95:
                command = closest_word[0]
                number_words = []
                total_count = 0
                for word in transcription.lower().split():
                    closest_number = process.extractOne(word, self.numbers_dict.keys(), scorer=fuzz.ratio)
                    if closest_number[1] >= threshold:
                        number_key = closest_number[0]
                        number_value = self.numbers_dict[number_key]
                        number_words.append(number_value['text'])
                        total_count += number_value['count']
                    elif word.isdigit():
                        count = int(word)
                        total_count += count
                        number_words.append(f"{count} вагонов")

                if number_words:
                    return f"{closest_word[0]} на {' '.join(number_words)}", self.count_commands_dict[
                        command], total_count
                else:
                    return closest_word[0], self.count_commands_dict[command], -1
        return "Команда не распознана", -1, -1

    def transcribe(self, audio_path: str) -> Dict[str, float]:
        """Транскрибирует аудиофайл.

        Args:
            audio_path: Путь к аудиофайлу.

        Returns:
            Словарь с результатами транскрибации, включая текст,
            скорость обработки и использование памяти.
        """
        start_time = time.time()

        # Загрузка аудио
        audio_data, _ = librosa.load(audio_path, sr=16000)

        # Подготовка входных данных
        input_values = self.processor(
            audio_data, sampling_rate=16000, return_tensors="np"
        ).input_values

        # Получение длительности аудиозаписи
        audio_duration = librosa.get_duration(y=audio_data, sr=16000)

        # Инференс
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_values}
        ort_outputs = self.ort_session.run(None, ort_inputs)

        # Постобработка
        logits = ort_outputs[0]
        predicted_ids = np.argmax(logits, axis=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        # Поиск ближайшей команды и атрибутов
        closest_command, label, attribute = self.find_closest_command(transcription)

        total_time = time.time() - start_time

        # Получение информации о использовании ресурсов
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "text": transcription,
            "processing_speed": total_time / audio_duration,
            "memory_usage": memory_info.rss / 1024 / 1024,  # МБ
            "label": label,
            "attribute": attribute,
            "closest_command": closest_command
        }
