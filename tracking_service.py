"""
Сервис для трекинга людей с использованием различных алгоритмов трекинга
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import ByteTrack, BotSort, DeepOcSort, OcSort, StrongSort
from pathlib import Path

from modules.logger import setup_logger
from modules.config import (
    RESOLUTION, MODEL_PATH, ANGLE_MIN, ANGLE_MAX, ANGLE_CENTER, SCREEN_BORDER,
    CONFIDENCE_THRESHOLD, POSE_MODEL, MAX_TRACK_HISTORY_SIZE, MAX_TRACKS_COUNT
)

logger = setup_logger(__name__)


class TrackingService:
    def __init__(self, model_path=POSE_MODEL):
        """
        Инициализация сервиса трекинга

        Args:
            model_path (str): Путь к модели YOLO
            tracker_name (str): Название алгоритма трекинга
        """
        try:
            self.tracker = self._initialize_tracker()
        except Exception as e:
            logger.error(f"Критическая ошибка при инициализации: {e}")
            raise

        # История трекинга
        self.track_history = {}
        # ID выбранного человека
        self.selected_track_id = None
        # Последняя точка
        self.last_point = np.array([RESOLUTION[0] // 2, RESOLUTION[1] // 2])
        # Текущие треки
        self.current_tracks = []
        # Счетчик кадров для периодической очистки памяти
        self.frame_counter = 0

    def _initialize_tracker(self):
        return ByteTrack(
            track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )

    def return_middle_point(self, frame, human_keypoints, draw=False):
        """Определение средней точки человека"""
        if human_keypoints.shape[0] == 0:
            return np.array([RESOLUTION[0] // 2, RESOLUTION[1] // 2])

        return np.array([RESOLUTION[0] // 2, RESOLUTION[1] // 2])

    @property
    def left_border(self):
        return int(RESOLUTION[0] * SCREEN_BORDER[0])

    @property
    def right_border(self):
        return int(RESOLUTION[0] * SCREEN_BORDER[1])

    def draw_borders(self, frame):
        """Отрисовка границ на кадре"""
        height, width, _ = frame.shape
        cv2.line(
            frame, (self.left_border, 0),
            (self.left_border, height), (255, 0, 0), 2
        )
        cv2.line(
            frame, (self.right_border, 0),
            (self.right_border, height), (255, 0, 0), 2
        )

    def get_current_tracks(self):
        """
        Получение текущих треков

        Returns:
            list: Список текущих треков
        """
        return self.current_tracks

    def get_pose_results(self):
        """
        Получение результатов распознавания поз

        Returns:
            list: Результаты распознавания поз
        """
        return self.last_results[0] if hasattr(self, 'last_results') else None

    def clear_gpu_memory(self):
        """Очистка кэша GPU для освобождения памяти"""
        try:
            # Очищаем неиспользуемую память GPU
            torch.cuda.empty_cache()
            logger.info("Очищен кэш GPU")
        except Exception as e:
            logger.error(f"Ошибка при очистке памяти GPU: {e}")

    def process_frame(self, frame):
        """
        Обработка кадра и трекинг людей

        Args:
            frame: Кадр изображения

        Returns:
            tuple: (обработанный кадр, координаты центра выбранного человека)
        """
        try:
            # Увеличиваем счетчик кадров
            self.frame_counter += 1

            # Периодически очищаем память GPU (например, каждые 100 кадров)
            if self.frame_counter % 100 == 0:
                self.clear_gpu_memory()
                # Логируем информацию о текущих треках в истории
                track_count = len(self.track_history)
                track_sizes = {tid: len(points) for tid, points in self.track_history.items()}
                total_points = sum(len(points) for points in self.track_history.values())
                logger.info(f"Статистика треков: всего={track_count}, точек={total_points}, размеры={track_sizes}")

            # Копируем кадр для отрисовки
            result_img = frame.copy()

            # Детекция объектов и поз
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(
                rgb_frame, classes=0, verbose=False, device='cuda', conf=CONFIDENCE_THRESHOLD
            )

            # Сохраняем результаты для последующего использования
            self.last_results = results
            logger.info(f"[DEBUG] TrackingService.process_frame: результаты: {results}")

            # Преобразование результатов YOLO для трекера
            boxes = []
            scores = []
            class_ids = []
            keypoints_list = []

            for result in results:
                # Получаем keypoints
                keypoints_cpu = [
                    keypoints.cpu() for keypoints in result.keypoints.xy
                ]

                # Получаем боксы и уверенность
                boxes.extend(result.boxes.xyxy.cpu().numpy())
                scores.extend(result.boxes.conf.cpu().numpy())
                class_ids.extend(result.boxes.cls.cpu().numpy().astype(int))
                keypoints_list.extend(keypoints_cpu)

            if boxes:
                # Преобразуем в numpy массивы
                boxes = np.array(boxes)
                scores = np.array(scores)
                class_ids = np.array(class_ids)

                # Формат для трекера
                tracker_input = np.column_stack((
                    boxes,
                    scores,
                    class_ids
                ))

                # Обновление трекера
                self.current_tracks = self.tracker.update(tracker_input, frame)

                # Если нет выбранного ID, выбираем первого человека
                if self.selected_track_id is None and len(self.current_tracks) > 0:
                    self.selected_track_id = int(self.current_tracks[0][4])

                # Обработка треков
                for track in self.current_tracks:
                    box = track[:4].astype(int)
                    track_id = int(track[4])
                    confidence = track[6]

                    # Получаем координаты центра
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Определяем цвет бокса
                    color = (0, 255, 0) if track_id == self.selected_track_id else (0, 0, 255)

                    # Рисуем бокс
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

                    # Добавляем ID и уверенность
                    label = f"ID: {track_id} ({confidence:.2f})"
                    cv2.putText(
                        result_img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    # Обновляем историю трекинга
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append((center_x, center_y))

                    # Ограничиваем историю трека максимальным размером
                    if len(self.track_history[track_id]) > MAX_TRACK_HISTORY_SIZE:
                        self.track_history[track_id] = self.track_history[track_id][-MAX_TRACK_HISTORY_SIZE:]

                    # Рисуем траекторию для выбранного человека
                    if track_id == self.selected_track_id:
                        points = self.track_history[track_id][-30:]
                        for j in range(1, len(points)):
                            cv2.line(
                                result_img,
                                points[j-1],
                                points[j],
                                (0, 0, 255),
                                2
                            )

                # Ограничиваем общее количество треков, если превышен лимит
                if len(self.track_history) > MAX_TRACKS_COUNT:
                    # Удаляем самые старые треки (исключая выбранный трек)
                    track_ids = list(self.track_history.keys())
                    track_ids = [tid for tid in track_ids if tid != self.selected_track_id]
                    tracks_to_remove = track_ids[:len(track_ids) - (MAX_TRACKS_COUNT - 1)]
                    for old_track_id in tracks_to_remove:
                        if old_track_id in self.track_history:
                            del self.track_history[old_track_id]

                # Получаем координаты центра выбранного человека
                if self.selected_track_id is not None:
                    selected_track = next(
                        (track for track in self.current_tracks if int(track[4]) == self.selected_track_id),
                        None
                    )
                    if selected_track is not None:
                        # Находим соответствующие keypoints для выбранного трека
                        # Предполагаем, что порядок треков совпадает с порядком в keypoints_list
                        box = selected_track[:4].astype(int)

                        # Ищем ближайший bounding box из исходных результатов
                        best_match_idx = -1
                        best_iou = 0

                        for i, result_box in enumerate(boxes):
                            # Вычисляем IoU (Intersection over Union)
                            x1 = max(box[0], result_box[0])
                            y1 = max(box[1], result_box[1])
                            x2 = min(box[2], result_box[2])
                            y2 = min(box[3], result_box[3])

                            if x2 < x1 or y2 < y1:
                                continue

                            intersection = (x2 - x1) * (y2 - y1)
                            area1 = (box[2] - box[0]) * (box[3] - box[1])
                            area2 = (result_box[2] - result_box[0]) * (result_box[3] - result_box[1])
                            union = area1 + area2 - intersection
                            iou = intersection / union if union > 0 else 0

                            if iou > best_iou:
                                best_iou = iou
                                best_match_idx = i

                        if best_match_idx >= 0 and best_match_idx < len(keypoints_list):
                            human_keypoints = keypoints_list[best_match_idx].numpy()
                            middle_point = self.return_middle_point(result_img, human_keypoints, draw=True)
                            return result_img, (int(middle_point[0]), int(middle_point[1]))

            return result_img, None

        except Exception as e:
            logger.error(f"Ошибка при обработке кадра: {e}")
            return frame, None

    def select_track(self, track_id):
        """
        Выбор конкретного человека для отслеживания

        Args:
            track_id (int): ID трека для выбора
        """
        logger.info(f"[DEBUG] TrackingService.select_track: Изменение ID трека с {self.selected_track_id} на {track_id}")
        # Проверяем, был ли трек с указанным ID обнаружен
        track_exists = any(int(track[4]) == track_id for track in self.current_tracks) if self.current_tracks is not None else False
        if not track_exists:
            logger.warning(f"[DEBUG] TrackingService.select_track: ВНИМАНИЕ! Трек с ID {track_id} не найден в текущих треках!")
            if self.current_tracks is not None:
                current_ids = [int(track[4]) for track in self.current_tracks]
                logger.warning(f"[DEBUG] TrackingService.select_track: Текущие доступные треки: {current_ids}")

        # Сохраняем предыдущий ID для отладки
        prev_id = self.selected_track_id
        self.selected_track_id = track_id
        logger.info(f"[DEBUG] TrackingService.select_track: После установки self.selected_track_id = {self.selected_track_id}")

        # Проверка успешности переключения
        if prev_id != self.selected_track_id:
            logger.info(f"[DEBUG] TrackingService.select_track: Успешное переключение с ID {prev_id} на ID {self.selected_track_id}")
        else:
            logger.warning(f"[DEBUG] TrackingService.select_track: ID трека не изменился: {self.selected_track_id}")
