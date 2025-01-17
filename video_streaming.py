import time
from numpy.ma.extras import average

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
Box_i_usr = 50
cropping_val = 0
factx = []
facty = []
usrednenie_flag = False

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
        self.spinBox_i_usr.valueChanged[int].connect(self.deBox_i_averaging)
        self.radioButton.clicked.connect(self.usrednen)
        self.spinBox_crop.valueChanged[int].connect(self.cropp)


        self.horizontalSlider_minR.valueChanged[int].connect(self.valueChangesminR)
        self.horizontalSlider_maxR.valueChanged[int].connect(self.valueChangesmaxR)
        self.horizontalSlider_param_1.valueChanged[int].connect(self.valueChanges_param_1)
        self.horizontalSlider_param_2.valueChanged[int].connect(self.valueChanges_param_2)
        # self.horizontalSlider_scale.valueChanged[int].connect(self.valueChangesScale)
        self.spinBox_scale.valueChanged[int].connect(self.valueChangesScale)



    def usrednen(self):
        global usrednenie_flag
        if self.radioButton.isChecked():
            usrednenie_flag = True
            self.radioButton.setText('усреднение'+'\n'+'включено')
        else:
            usrednenie_flag = False
            self.radioButton.setText('усреднение' + '\n' + 'отключено')
            self.factx_TEXT.setText('FactX')
            self.factY_TEXT.setText('FactY')

    def cropp(self, val_crop):
        global cropping_val
        cropping_val = int(val_crop)


    def deBox_i_averaging(self, val_i):
        global Box_i_usr
        Box_i_usr = val_i
        self.label_i_usr.setText('iуср' + '\n' + 't=' + str(Box_i_usr * 0.05)[:3])

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
        # print('real_radius', real_radius)

    def linePort_2(self):

        teeeext = str(self.lineEdit_2.text()).split(':')


        # print(teeeext)
        global port1
        port1 = int(teeeext[1])

        global ipAdr
        ipAdr = str(teeeext[0])

        # print('ip', ipAdr,'port', port1)

        # global port1
        # port1 = int(self.lineEdit_2.text()[-3:])
        #
        # global ipAdr
        # ipAdr = self.lineEdit_2.text()[:11]
        #
        # print('ip,port', ipAdr, port1)

        # port1some = int(self.lineEdit_2.text())
        # print(str(port1some))

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
        # print(value5, scale)
        # self.label_6.setText('scale ' + str(scale)[:4])



    def init_properties(self):
        self.stream_thread = Stream_thread()
        self.Modbus_C_S = Modbus_Client_Server()


    def init_connections(self):
        self.stream_thread.change_pixmap.connect(self.image_label.setPixmap)
        self.start_stop_btn.clicked.connect(self.run_stop_video_streaming)
        self.radioButtonMODBUS_START.clicked.connect(self.start_Modbus)

    @QtCore.pyqtSlot(bool)
    def start_Modbus(self):
        if self.radioButtonMODBUS_START.isChecked():
            self.Modbus_C_S.start()
            self.radioButtonMODBUS_START.setText('Started')
        else:
            self.Modbus_C_S.stop_Modbus()
            self.radioButtonMODBUS_START.setText('Stopped')


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




class Modbus_Client_Server(QtCore.QThread):


    def run(self):
        oldMW3 = 0
        self.averXnumber = 0
        self.averYnumber = 0
        self.isConnectedMod = 0
        self.PLC1 = MODBUS_TCP_master()

        self.isConnectedMod = self.PLC1.Start_TCP_client(IP_address=ipAdr, TCP_port=port1)


        self.thread_is_active_MODBUS = True
        if self.isConnectedMod == 1:

            while self.thread_is_active_MODBUS:
                MW3 = self.PLC1.Read_holding_register_uint16(Register_address=BoxRX)

                if oldMW3 != MW3:
                    if usrednenie_flag:
                        averageX = []
                        averageY = []
                        if factx and facty:
                            self.averXnumber = factx[0]
                            self.averYnumber = facty[0]


                        for usr in range(Box_i_usr):
                            if factx and facty and (self.averXnumber - 5 <= factx[0] <= self.averXnumber + 5) and (self.averYnumber - 5 <= facty[0] <= self.averYnumber + 5):
                                averageX.append(factx[0])
                                self.averXnumber = sum(averageX) / len(averageX)
                                w.factx_TEXT.setText('factX:' + str(self.averXnumber)[:5])

                                averageY.append(facty[0])
                                self.averYnumber = sum(averageY) / len(averageY)
                                w.factY_TEXT.setText('factY:'+ str(self.averYnumber)[:5])
                                time.sleep(0.05)

                            else:
                                time.sleep(0.05)

                        # print(self.averXnumber,len(averageX), averageX)
                        # print(self.averYnumber, len(averageY), averageY)

                        self.PLC1.Write_multiple_holding_register_float32(Register_address=BoxX, Register_value=self.averXnumber)
                        self.PLC1.Write_multiple_holding_register_float32(Register_address=BoxY,Register_value=self.averYnumber)
                        self.PLC1.Write_multiple_holding_register_uint16(Register_address=BoxLEN,Register_value=len(factx))

                    else:
                        if factx and facty:
                            self.PLC1.Write_multiple_holding_register_float32(Register_address=BoxX, Register_value=factx[0])
                            self.PLC1.Write_multiple_holding_register_float32(Register_address=BoxY, Register_value=facty[0])
                            self.PLC1.Write_multiple_holding_register_uint16(Register_address=BoxLEN, Register_value=len(factx))
                    oldMW3 = MW3
        else:
            w.radioButtonMODBUS_START.setText('нет подключения ' + str(ipAdr) + ':' + str(port1))



    def stop_Modbus(self):
        self.thread_is_active_MODBUS = False
        # self.PLC1.Stop_TCP_client()

        if self.isConnectedMod == 1:
            self.PLC1.Stop_TCP_client_ChutChut()

        self.quit()



class Stream_thread(QtCore.QThread, Ui_MainWindow):

    change_pixmap = QtCore.pyqtSignal(QtGui.QPixmap)

    # def median_smoothing(self, current, previous):
    #     if previous is None:
    #         return current
    #     return tuple(int(np.median([current[i], previous[i]])) for i in range(len(current)))


    def CirclesCenters(self, image):

        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = cv2.GaussianBlur(im, (9, 9), 2)  # Применение Гауссового размытия
        rows = im.shape[0]
        circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, int(rows / 8),
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

    def incr_koord_dp(self, img, l, real_radius):

        # пикселей на мм по оси х и у
        # расчет абсолютных координат
        x123 = []
        y123 = []
        radius_list = []
        for i in range(len(l)):
            x123.append(l[i][0])
            y123.append(l[i][1])
            radius_list.append(l[i][2])


        # вычисление новых коорд относительно центра кадра

        h, w = img.shape[:2]

        nolx = w // 2
        noly = h // 2

        # image = cv2.circle(img, (nolx, noly), 1, (255, 0, 0), 5)
        cv2.line(img, (nolx, 0),(nolx, h),  (255, 0, 0), 2)
        cv2.line(img, (0, noly), (w, noly), (255, 0, 0), 2)

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

    def show_koords(self):

        if factx and facty:
            w.factX.setText(str(factx))
            w.factY.setText(str(facty))




    def run(self):
        # Инициализация предыдущих координат и радиусов

        self.prev_center_koord = []
        self.prev_radius_list = []
        self.smoothing_factor = 0.5  # Параметр сглаживания
        # self.position_threshold = 10  # Порог для устранения выбросов

        cap = cv2.VideoCapture(comportCAM)
        self.thread_is_active = True

        while self.thread_is_active:
            ret, image = cap.read()
            if ret:

                # Проверяем, достаточно ли размер кадра для обрезки
                if (cropping_val > 1) and (image.shape[0] > 2 * cropping_val) and (image.shape[1] > 2 * cropping_val):
                    # Обрезаем кадр равномерно со всех сторон
                    image = image[cropping_val:image.shape[0] - cropping_val, cropping_val:image.shape[1] - cropping_val]

                #     # Если кадр слишком мал для обрезки, просто оставляем его как есть

                image, center_koord = self.CirclesCenters(image)

                global factx, facty
                factx, facty = self.incr_koord_dp(image, center_koord, real_radius)

                self.thr1 = threading.Thread(target=self.show_koords, daemon=True).start()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                qt_image = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format_RGB888)
                pic = qt_image.scaled(int(image.shape[1]*scale), int(image.shape[0]*scale), QtCore.Qt.KeepAspectRatio)
                pixmap = QtGui.QPixmap.fromImage(pic)
                self.change_pixmap.emit(pixmap)


    def stop(self):
        self.thread_is_active = False
        self.quit()

       


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()

    sys.exit(app.exec_())
