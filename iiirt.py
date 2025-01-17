import cv2
import numpy as np

class YourClass:
    def __init__(self):
        self.prev_center_koord = []
        self.prev_radius_list = []
        self.smoothing_factor = 0.5

    def CirclesCenters1(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = cv2.GaussianBlur(im, (9, 9), 2)  # Применение Гауссового размытия
        rows = im.shape[0]
        circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, BoxDP, rows / 8,
                                   param1=param_1, param2=param_2,
                                   minRadius=minR, maxRadius=maxR)
        center_koord = []

        current_centers = []

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                center = (i[0], i[1], i[2])  # (x, y, radius)

                # Фильтрация по радиусу
                if minR <= center[2] <= maxR:
                    current_centers.append(center)

                    # Сглаживание координат и радиуса
                    if len(self.prev_center_koord) > 0:
                        # Найти ближайшую предыдущую окружность
                        distances = [np.linalg.norm(np.array(center[0:2]) - np.array(prev[0:2])) for prev in self.prev_center_koord]
                        closest_index = np.argmin(distances)
                        closest_distance = distances[closest_index]

                        # Убедитесь, что ближайшая окружность достаточно близка
                        if closest_distance < 50:  # Пороговое значение для расстояния
                            closest_prev = self.prev_center_koord[closest_index]

                            center = (
                                int(self.smoothing_factor * center[0] + (1 - self.smoothing_factor) * closest_prev[0]),
                                int(self.smoothing_factor * center[1] + (1 - self.smoothing_factor) * closest_prev[1]),
                                int(self.smoothing_factor * center[2] + (1 - self.smoothing_factor) * closest_prev[2])
                            )

                    # Отрисовка центра и окружности
                    image = cv2.circle(image, center[0:-1], 1, (0, 0, 255), 3)
                    image = cv2.circle(image, center[0:-1], center[-1], (0, 250, 0), 2)
                    center_koord.append(list(map(int, center)))

            center_koord = sorted(center_koord)

            if len(center_koord) != 0:
                image = cv2.circle(image, (center_koord[0][0], center_koord[0][1]), center_koord[0][2], (255, 0, 0), 5)

            # Обновляем предыдущие значения
            self.prev_center_koord = current_centers
            self.prev_radius_list = [c[2] for c in current_centers]
        else:
            # Если окружности не найдены, сбрасываем предыдущие значения
            self.prev_center_koord = []
            self.prev_radius_list = []

        return image, center_koord






    def CirclesCenters(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows = im.shape[0]
        circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, BoxDP, rows / 8,
                                   param1=param_1, param2=param_2,
                                   minRadius=minR, maxRadius=maxR)
        center_koord = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1], i[2])
                image = cv2.circle(image, center[0:-1], 1, (0, 0, 255), 3)
                image = cv2.circle(image, center[0:-1], center[-1], (0, 250, 0), 2)
                center_koord.append(list(map(int, center)))
        center_koord = sorted(center_koord)
        if len(center_koord) != 0:
            image = cv2.circle(image, (center_koord[0][0:-1]), center_koord[0][-1], (255, 0, 0), 5)
        return image, center_koord











    def __init__(self):
        self.prev_center_koord = []
        self.prev_radius_list = []
        self.smoothing_factor = 0.5
        self.position_threshold = 10  # Порог для устранения выбросов




    def median_smoothing(self, current, previous):
        if previous is None:
            return current
        return tuple(int(np.median([current[i], previous[i]])) for i in range(len(current)))




    def CirclesCenters(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = cv2.GaussianBlur(im, (9, 9), 2)
        rows = im.shape[0]
        circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, BoxDP, rows / 8,
                                   param1=param_1, param2=param_2,
                                   minRadius=minR, maxRadius=maxR)
        center_koord = []

        current_centers = []

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                center = (i[0], i[1], i[2])  # (x, y, radius)

                # Фильтрация по радиусу
                if minR <= center[2] <= maxR:
                    current_centers.append(center)

                    # Сглаживание координат и радиуса
                    if len(self.prev_center_koord) > 0:
                        # Найти ближайшую предыдущую окружность
                        distances = [np.linalg.norm(np.array(center[0:2]) - np.array(prev[0:2])) for prev in
                                     self.prev_center_koord]
                        closest_index = np.argmin(distances)
                        closest_distance = distances[closest_index]

                        # Убедитесь, что ближайшая окружность достаточно близка
                        if closest_distance < 50:  # Пороговое значение для расстояния
                            closest_prev = self.prev_center_koord[closest_index]

                            # Устранение выбросов
                            if (abs(center[0] - closest_prev[0]) < self.position_threshold and
                                    abs(center[1] - closest_prev[1]) < self.position_threshold):
                                center = self.median_smoothing(center, closest_prev)

                    # Отрисовка центра и окружности
                    image = cv2.circle(image, center[0:-1], 1, (0, 0, 255), 3)
                    image = cv2.circle(image, center[0:-1], center[-1], (0, 250, 0), 2)
                    center_koord.append(list(map(int, center)))

            center_koord = sorted(center_koord)

            if len(center_koord) != 0:
                image = cv2.circle(image, (center_koord[0][0], center_koord[0][1]), center_koord[0][2], (255, 0, 0), 5)

                # Обновляем предыдущие значения
            self.prev_center_koord = current_centers
            self.prev_radius_list = [c[2] for c in current_centers]
        else:
            # Если окружности не найдены, сбрасываем предыдущие значения
            self.prev_center_koord = []
            self.prev_radius_list = []

        return image, center_koord






