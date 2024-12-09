from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout,
                             QGroupBox, QMenu, QPushButton,
                             QRadioButton, QVBoxLayout,
                             QWidget, QSlider, QLabel, QMainWindow)
import sys


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        uic.loadUi('video_streaming.ui', self)
        self.setWindowTitle("SLIDER")
        self.slider =  self.findChild(QSlider, "horizontalSlider_param_1")
        self.label = self.findChild(QLabel, "label_3")
        self.slider.valueChanged.connect(self.slide_it)


        self.show()
    def slide_it(self, value):
        self.label.setText(str(value))


app = QApplication
UIWindow = UI()
app.exec_()
