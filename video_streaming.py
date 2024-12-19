import time

from sys import flags

import sys
import numpy as np
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from numpy.ma.core import resize

from UI_video_streaming import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from MODBUS_TCP_CLIENT import *

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout,
                             QGroupBox, QMenu, QPushButton,
                             QRadioButton, QVBoxLayout,
                             QWidget, QSlider, QLabel, QMainWindow)



import threading








minR = maxR = 0
param_1 = param_2 = 100
scale = 0.9
port1 = 502
# ipAdr = '192.168.1.8'
ipAdr = '127.0.0.1'
real_radius = 10
comportCAM = 0
BoxX = 10
BoxRX = 20
BoxY = 30
BoxLEN = 40
BoxDP = 1.00
factx = []
facty = []

# stop_thread_update = False
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, ):
        super(MainWindow, self).__init__(parent=parent)

        self.setupUi(self)
        self.init_properties()
        self.init_connections()

        self.lineEdit_2.returnPressed.connect(self.linePort_2)
        self.lineEdit.returnPressed.connect(self.lineRadius)
        self.spinBoxCOMPORT.valueChanged[int].connect(self.comID)
        self.spinBoxX.valueChanged[int].connect(self.deBoxX)
        self.spinBoxY.valueChanged[int].connect(self.deBoxY)
        self.spinBoxRX.valueChanged[int].connect(self.deBoxRX)
        self.spinBoxLEN.valueChanged[int].connect(self.deBoxLEN)
        self.doubleSpinBoxdp.valueChanged[float].connect(self.deBexDP)


        self.horizontalSlider_minR.valueChanged[int].connect(self.valueChangesminR)
        self.horizontalSlider_maxR.valueChanged[int].connect(self.valueChangesmaxR)
        self.horizontalSlider_param_1.valueChanged[int].connect(self.valueChanges_param_1)
        self.horizontalSlider_param_2.valueChanged[int].connect(self.valueChanges_param_2)
        self.horizontalSlider_scale.valueChanged[int].connect(self.valueChangesScale)



        thr1 = threading.Thread(target=self.myfunc, daemon=True).start()



    def myfunc(self):
        global factx, facty
        while True:
            self.factX.setText(str(factx))
            self.factY.setText(str(facty))
            time.sleep(1)

    def deBexDP(self, valDP):
        global BoxDP
        BoxDP = valDP

    def deBoxLEN(self, valLEN):
        global BoxLEN
        BoxLEN = valLEN

    def deBoxRX(self, valRX):
        global BoxRX
        BoxRX = valRX

    def deBoxY(self, valY):
        global BoxY
        BoxY = valY

    def deBoxX(self, valX):
        global BoxX
        BoxX = valX

    def comID(self,valu):
        global comportCAM
        comportCAM = valu

    def lineRadius(self):
        global real_radius
        real_radius = float(self.lineEdit.text())
        print('real_radius', real_radius)

    def linePort_2(self):
        global port1
        port1 = int(self.lineEdit_2.text()[-3:])


        global ipAdr
        ipAdr = self.lineEdit_2.text()[:9]

        print('ip,port', ipAdr, port1)

    def valueChangesminR(self, value1):
        global minR
        minR = value1
        self.label_5.setText('minR ' + str(minR))

    def valueChangesmaxR(self, value2):
        global maxR
        maxR = value2
        self.label_2.setText('maxR ' + str(maxR))

    def valueChanges_param_1(self, value3):
        global param_1
        param_1 = value3
        self.label_3.setText('param1 ' + str(param_1))

    def valueChanges_param_2(self, value4):
        global param_2
        param_2 = value4
        self.label_4.setText('param2 ' + str(param_2))

    def valueChangesScale(self, value5):
        global scale
        scale = value5/500
        self.label_6.setText('scale ' + str(scale))



    def init_properties(self):
        self.stream_thread = Stream_thread()



    def init_connections(self):
        self.stream_thread.change_pixmap.connect(self.image_label.setPixmap)
        self.start_stop_btn.clicked.connect(self.run_stop_video_streaming)



    @QtCore.pyqtSlot(bool)
    def run_stop_video_streaming(self):
        if self.start_stop_btn.isChecked():
            self.stream_thread.start()
            self.update_button_style()

        else:
            self.stream_thread.stop()
            self.update_button_style()







    
    def update_button_style(self):
        if self.start_stop_btn.isChecked():
            icon_stop = QtGui.QIcon()
            icon_stop.addPixmap(QtGui.QPixmap(":/icons/icons/stop_video.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.start_stop_btn.setIcon(icon_stop)
            self.start_stop_btn.setStyleSheet("border: 2px solid red; border-radius: 7px;")

        else:
            icon_run = QtGui.QIcon()
            icon_run.addPixmap(QtGui.QPixmap(":/icons/icons/run_video.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.start_stop_btn.setIcon(icon_run)
            self.start_stop_btn.setStyleSheet("border: none solid blue; border-radius: 7px;")








class Stream_thread(QtCore.QThread, Ui_MainWindow):

    change_pixmap = QtCore.pyqtSignal(QtGui.QPixmap)



    def CirclesCenters(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows = im.shape[0]
        #param1 Верхний порог для детектора краёв
        #param2 порог для центра окружностей в накопителе
        circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, BoxDP, rows / 8,
                                   param1=param_1, param2=param_2,
                                   minRadius=minR, maxRadius=maxR)
        center_koord = []
        radius_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                center_koord.append(list(map(int, center)))

                # circle center
                image = cv2.circle(image, center, 1, (0, 0, 255), 3)
                # circle outline
                radius = i[2]
                image = cv2.circle(image, center, radius, (0, 250, 0), 2)
                radius_list.append(int(i[2]))

            # print('radius_list!!!!!', radius_list)
            # print('center_koord!!!!!', center_koord)

        # return center_koord, radius_list
        return image, center_koord, radius_list

    def incr_koord_dp(self, img, l, radius_list, real_radius):

        # пикселей на мм по оси х и у
        # расчет абсолютных координат
        x123 = []
        y123 = []
        for i in range(len(l)):
            x123.append(l[i][0])
            y123.append(l[i][1])


        # вычисление новых коорд относительно центра кадра

        (h, w) = img.shape[:2]

        nolx = w // 2
        noly = h // 2

        # image = cv2.circle(img, (nolx, noly), 1, (255, 0, 0), 5)
        image = cv2.line(img, (nolx, 0),(nolx, h),  (255, 0, 0), 1)
        image = cv2.line(img, (0, noly), (w, noly), (255, 0, 0), 1)

        for i in range(len(x123)):
            x123[i] = x123[i] - nolx
            y123[i] = y123[i] - noly


        factx = []
        facty = []
        d_real_r = []
        mm_nolx = mm_noly = 0

        rast_o_dot_p = []
        rast_o_dot_mm = []

        if real_radius != 0 and radius_list:

            for rr in range(len(radius_list)):
                d_real_r.append(real_radius / radius_list[rr])

                factx.append(x123[rr] * d_real_r[rr])
                facty.append(y123[rr] * d_real_r[rr])

            for itr in range(len(x123)):
                mm_nolx = ((sum(d_real_r) / len(d_real_r)) * nolx)
                mm_noly = ((sum(d_real_r) / len(d_real_r)) * noly)

                rast_o_dot_p.append((((x123[itr]) ** 2) + ((y123[itr]) ** 2)) ** 0.5)

                rast_o_dot_mm.append((((factx[itr]) ** 2) + ((facty[itr]) ** 2)) ** 0.5)


            # print('\n')
            # print('Коэффициент отношения мм к пикселям', d_real_r)
            # print('x координаты в мм отн. центра кадра', factx)
            # print('y координаты в мм отн. центра кадра', facty)
            #
            # print('координаты центра кадра в мм       ', mm_nolx, mm_noly)
            # print('расстояние до центров в пикселях   ', rast_o_dot_p)
            # print('расстояние до центров в мм         ', rast_o_dot_mm,'\n')

        else:
            facty = []
            factx = []

        return factx, facty



    def run(self):
        oldMW3 = 0
        flagCon = 0
        self.PLC1 = MODBUS_TCP_master()
        self.PLC1.Start_TCP_client(IP_address=ipAdr, TCP_port = port1)



        cap = cv2.VideoCapture(comportCAM)
        self.thread_is_active = True

        # print("параметры сервера:",ipAdr, port1)
        # print('порты:', 'x', BoxX, 'y', BoxY, 'rx', BoxRX, 'len', BoxLEN)
        # print("minR", minR, "maxR", maxR, "param_1", param_1, "param_2", param_2)

        while self.thread_is_active:
            MW3 = self.PLC1.Read_holding_register_uint16(Register_address=BoxRX)



            ret, image = cap.read()
            if ret:
                image, center_koord, radius_list = self.CirclesCenters(image)
                global factx, facty
                factx, facty = self.incr_koord_dp(image, center_koord, radius_list, real_radius)



                # print('factxxxxxxxx, facty', str(factx), str(facty))
                #image = self.resized(image)

                # print('!!!!!!!!!!!', center_koord, radius_list)
                # image = self.rotation(image, 30)
                # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # flipped_image = cv2.flip(image, 1)

                qt_image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_BGR888)
                pic = qt_image.scaled(int(image.shape[1]*scale), int(image.shape[0]*scale), QtCore.Qt.KeepAspectRatio)
                pixmap = QtGui.QPixmap.fromImage(pic)
                self.change_pixmap.emit(pixmap)



                if MW3 in range(len(factx)):
                    if oldMW3 != MW3 and factx:
                        self.PLC1.Write_multiple_holding_register_float32(Register_address=BoxX, Register_value=factx[MW3])
                        self.PLC1.Write_multiple_holding_register_float32(Register_address=BoxY, Register_value=facty[MW3])
                        self.PLC1.Write_multiple_holding_register_uint16(Register_address=BoxLEN, Register_value=len(factx))
                        oldMW3 = MW3




    def stop(self):
        self.PLC1.Stop_TCP_client()
        self.thread_is_active = False

        self.quit()

       
         
# def main():
#    app = QApplication(sys.argv)
#
#    ex.show()
#    sys.exit(app.exec_())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())