import sys
import numpy as np
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from numpy.ma.core import resize

from UI_video_streaming import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout,
                             QGroupBox,QMenu, QPushButton,
                             QRadioButton, QVBoxLayout,
                             QWidget, QSlider,QLabel)



class sliderdemo(QWidget):
    def __init__(self, vSl=90, parent=None):
        super(sliderdemo, self).__init__(parent)
        layout = QVBoxLayout()
        self.sl = QSlider(Qt.Horizontal)
        # self.sl.setMinimum(0)
        # self.sl.setMaximum(180)
        # self.sl.setValue(vSl)
        # self.sl.setTickPosition(QSlider.TicksBelow)
        # self.sl.setTickInterval(10)

        layout.addWidget(self.sl)
        self.sl.valueChanged[int].connect(self.valuechange)
        self.setLayout(layout)
        self.setWindowTitle("slider")
        print(self.valuechange(self.sl.valueChanged))
        print("__init__vSl -> ", vSl)


    def valuechange(self, value):
        #self.size = self.sl.value()
        self.__init__(value)
        #return self.size



def main():
   app = QApplication(sys.argv)
   ex = sliderdemo(90)
   ex.show()
   sys.exit(app.exec_())



if __name__ == '__main__':
   main()