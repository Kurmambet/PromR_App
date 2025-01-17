# import sys
# import cv2
# from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtCore import QTimer
#
# class VideoCapture(QWidget):
#     def __init__(self, crop_value):
#         super().__init__()
#         self.setWindowTitle("Video Capture")
#         self.setGeometry(100, 100, 800, 600)
#
#         self.label = QLabel(self)
#         layout = QVBoxLayout()
#         layout.addWidget(self.label)
#         self.setLayout(layout)
#
#         self.cap = cv2.VideoCapture(0)
#         self.crop_value = crop_value
#
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(30)  # обновление каждые 30 мс
#
#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if ret:
#             # Обрезаем кадр
#             height, width, _ = frame.shape
#             frame = frame[self.crop_value:height-self.crop_value, self.crop_value:width-self.crop_value]
#
#             # Конвертируем BGR в RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Преобразуем в QImage
#             h, w, ch = frame.shape
#             bytes_per_line = ch * w
#             q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
#
#             # Обновляем QLabel
#             self.label.setPixmap(QPixmap.fromImage(q_img))
#
#     def closeEvent(self, event):
#         self.cap.release()
#         event.accept()
#
# if __name__ == "__main__":
#     crop_value = 1  # Задайте значение обрезки
#     app = QApplication(sys.argv)
#     window = VideoCapture(crop_value)
#     window.show()
#     sys.exit(app.exec_())


# import sys
# import cv2
# from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtCore import QTimer
#
#
# class VideoCapture(QWidget):
#     def __init__(self, crop_value):
#         super().__init__()
#         self.setWindowTitle("Video Capture")
#         self.setGeometry(100, 100, 800, 600)
#
#         self.label = QLabel(self)
#         layout = QVBoxLayout()
#         layout.addWidget(self.label)
#         self.setLayout(layout)
#
#         self.cap = cv2.VideoCapture(0)
#         self.crop_value = crop_value
#
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(30)  # обновление каждые 30 мс
#
#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if ret:
#             height, width, _ = frame.shape
#
#             # Проверяем, достаточно ли размер кадра для обрезки
#             if height > 2 * self.crop_value and width > 2 * self.crop_value:
#                 # Обрезаем кадр равномерно со всех сторон
#                 frame = frame[self.crop_value:height - self.crop_value, self.crop_value:width - self.crop_value]
#             else:
#                 # Если кадр слишком мал для обрезки, просто оставляем его как есть
#                 frame = frame
#
#             # Конвертируем BGR в RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Преобразуем в QImage
#             h, w, ch = frame.shape
#             bytes_per_line = ch * w
#             q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
#
#             # Обновляем QLabel
#             self.label.setPixmap(QPixmap.fromImage(q_img))
#
#     def closeEvent(self, event):
#         self.cap.release()
#         event.accept()
#
#
# if __name__ == "__main__":
#     crop_value = 10  # Задайте значение обрезки
#     app = QApplication(sys.argv)
#     window = VideoCapture(crop_value)
#     window.show()
#     sys.exit(app.exec_())


# import sys
# import cv2
# from PyQt5 import QtGui, QtCore
# from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
# from PyQt5.QtCore import QTimer


# class VideoCapture(QWidget):
#     change_pixmap = QtCore.pyqtSignal(QtGui.QPixmap)
#
#     def __init__(self, crop_value):
#         super().__init__()
#         self.setWindowTitle("Video Capture")
#         self.setGeometry(100, 100, 800, 600)
#
#         self.label = QLabel(self)
#         layout = QVBoxLayout()
#         layout.addWidget(self.label)
#         self.setLayout(layout)
#
#         self.cap = cv2.VideoCapture(0)
#         self.crop_value = crop_value
#
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(30)  # обновление каждые 30 мс
#
#         self.change_pixmap.connect(self.update_image)
#
#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if ret:
#             height, width, _ = frame.shape
#
#             # Обрезаем кадр равномерно со всех сторон
#             if height > 2 * self.crop_value and width > 2 * self.crop_value:
#                 frame = frame[self.crop_value:height - self.crop_value, self.crop_value:width - self.crop_value]
#
#             # Преобразуем BGR в RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Создаем QImage
#             qt_image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
#                                     QtGui.QImage.Format_RGB888)
#
#             # Эмитируем сигнал с QPixmap
#             pixmap = QtGui.QPixmap.fromImage(qt_image)
#             self.change_pixmap.emit(pixmap)
#
#     def update_image(self, pixmap):
#         self.label.setPixmap(pixmap)
#
#     def closeEvent(self, event):
#         self.cap.release()
#         event.accept()
#
#
# if __name__ == "__main__":
#     crop_value = 50  # Задайте значение обрезки
#     app = QApplication(sys.argv)
#     window = VideoCapture(crop_value)
#     window.show()
#     sys.exit(app.exec_())

#
# import sys
# from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
# from PyQt5.QtGui     import QPixmap, QImage, qRgb
# from PyQt5.QtCore    import Qt
#
# import numpy as np
# import cv2
#
# app = QApplication(sys.argv)
#
# gray_color_table = [qRgb(i, i, i) for i in range(256)]
#
# def NumpyToQImage(im):
#     qim = QImage()
#     if im is None:
#         return qim
#     if im.dtype == np.uint8:
#         if len(im.shape) == 2:
#             qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
#             qim.setColorTable(gray_color_table)
#         elif len(im.shape) == 3:
#             if im.shape[2] == 3:
#                 qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
#             elif im.shape[2] == 4:
#                 qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
#     return qim
#
# img = cv2.imread('D:/programirovanie/NEWPYTHONPROJECT/.venv/processPhotoess_302.jpg')
# qimg = NumpyToQImage(img)
# assert(not qimg.isNull())
#
# label = QLabel()
# pixmap = QPixmap(qimg)
# pixmap.scaled(200, 200)
# label.setPixmap(pixmap)
# label.show()
#
# label_1 = QLabel()
# pixmap  = QPixmap('D:/programirovanie/NEWPYTHONPROJECT/.venv/processPhotoess_302.jpg')
# label_1.setPixmap(pixmap)
# label_1.setScaledContents(True)
# label_1.show()
#
# img = cv2.imread('D:/programirovanie/NEWPYTHONPROJECT/.venv/processPhotoess_302.jpg')
# img = np.copy(img[0:90, 140:224, :])
# qimg2 = NumpyToQImage(img)
# assert(not qimg2.isNull())
#
# label_2 = QLabel()
# label_2.resize(224, 224)
# pixmap = QPixmap(qimg2)
# pixmap.scaled(300, 300, Qt.IgnoreAspectRatio, Qt.FastTransformation)
# label_2.setPixmap(pixmap)
# label_2.setScaledContents(True)
# label_2.show()
#
# app.exec_()


# import sys
# import cv2
# from PyQt5 import QtGui, QtCore
# from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QSlider, QHBoxLayout
# from PyQt5.QtCore import QTimer
#
#
# class VideoCapture(QWidget):
#     change_pixmap = QtCore.pyqtSignal(QtGui.QPixmap)
#
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Video Capture")
#         self.setGeometry(100, 100, 800, 600)
#
#         self.label = QLabel(self)
#         layout = QVBoxLayout()
#         layout.addWidget(self.label)
#
#         # Создаем ползунок для изменения crop_value
#         self.slider = QSlider(QtCore.Qt.Horizontal, self)
#         self.slider.setRange(0, 200)  # Установите диапазон значений
#         self.slider.setValue(50)  # Начальное значение
#         self.slider.valueChanged.connect(self.update_crop_value)
#         layout.addWidget(self.slider)
#
#         self.setLayout(layout)
#
#         self.cap = cv2.VideoCapture(0)
#         self.crop_value = self.slider.value()  # Инициализируем crop_value
#
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(30)  # обновление каждые 30 мс
#
#         self.change_pixmap.connect(self.update_image)
#
#     def update_crop_value(self, value):
#         self.crop_value = value  # Обновляем значение crop_value
#
#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if ret:
#             height, width, _ = frame.shape
#
#             # Обрезаем кадр равномерно со всех сторон
#             if height > 2 * self.crop_value and width > 2 * self.crop_value:
#                 frame = frame[self.crop_value:height - self.crop_value, self.crop_value:width - self.crop_value]
#
#             # Преобразуем BGR в RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Создаем QImage
#             qt_image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
#                                     QtGui.QImage.Format_RGB888)
#
#             # Эмитируем сигнал с QPixmap
#             pixmap = QtGui.QPixmap.fromImage(qt_image)
#             self.change_pixmap.emit(pixmap)
#
#     def update_image(self, pixmap):
#         self.label.setPixmap(pixmap)
#
#     def closeEvent(self, event):
#         self.cap.release()
#         event.accept()
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = VideoCapture()
#     window.show()
#     sys.exit(app.exec_())


import sys
import cv2
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QSlider


class VideoCaptureThread(QtCore.QThread):
    change_pixmap = QtCore.pyqtSignal(QtGui.QPixmap)

    def __init__(self):
        super().__init__()
        self.running = True
        self.crop_value = 50  # Начальное значение crop_value

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                height, width, _ = frame.shape

                # Обрезаем кадр равномерно со всех сторон
                if height > 2 * self.crop_value and width > 2 * self.crop_value:
                    frame = frame[self.crop_value:height - self.crop_value, self.crop_value:width - self.crop_value]

                # Преобразуем BGR в RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Создаем QImage
                qt_image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                                        QtGui.QImage.Format_RGB888)

                # Эмитируем сигнал с QPixmap
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                self.change_pixmap.emit(pixmap)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def set_crop_value(self, value):
        self.crop_value = value


class VideoCapture(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Capture")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        # Создаем ползунок для изменения crop_value
        self.slider = QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, 200)  # Установите диапазон значений
        self.slider.setValue(50)  # Начальное значение
        self.slider.valueChanged.connect(self.update_crop_value)
        layout.addWidget(self.slider)

        self.setLayout(layout)

        # Создаем и запускаем поток для захвата видео
        self.video_thread = VideoCaptureThread()
        self.video_thread.change_pixmap.connect(self.update_image)
        self.video_thread.start()

    def update_crop_value(self, value):
        self.video_thread.set_crop_value(value)  # Обновляем значение crop_value в потоке

    def update_image(self, pixmap):
        self.label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.video_thread.stop()  # Останавливаем поток при закрытии
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoCapture()
    window.show()
    sys.exit(app.exec_())




# desired_aspect_ratio = 16 / 9  # Например, 16:9
# current_aspect_ratio = width / height
#
# if current_aspect_ratio > desired_aspect_ratio:
#     # Изображение шире, чем нужно, обрезаем по ширине
#     new_width = height * desired_aspect_ratio
#     crop_value = (width - new_width) / 2
#     frame = frame[self.crop_value:height-self.crop_value, int(crop_value):int(width-crop_value)]
# else:
#     # Изображение уже, чем нужно, обрезаем по высоте
#     new_height = width / desired_aspect_ratio
#     crop_value = (height - new_height) / 2
#     frame = frame[int(crop_value):int(height-crop_value), self.crop_value:width-self.crop_value]






