import sys
import threading
import time

import cv2
import numpy as np
from MODBUS_TCP_CLIENT import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from UI_video_streaming import Ui_MainWindow

minR = maxR = 0
param_1 = param_2 = 100
scale = 0.9
port1 = 502
# ipAdr = '192.168.1.8'
ipAdr = "127.0.0.1"
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

# Мьютекс для безопасного доступа к factx/facty из двух потоков
data_lock = threading.Lock()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
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
        self.horizontalSlider_param_1.valueChanged[int].connect(
            self.valueChanges_param_1
        )
        self.horizontalSlider_param_2.valueChanged[int].connect(
            self.valueChanges_param_2
        )

        self.spinBox_scale.valueChanged[int].connect(self.valueChangesScale)
        self.pushButton_open.clicked.connect(self.open_file)
        self.pushButton_save.clicked.connect(self.save_file)

    def save_file(self):
        saving_data = {
            "minR": minR,
            "maxR": maxR,
            "param_1": param_1,
            "param_2": param_2,
            "scale": scale,
            "port1": port1,
            "ipAdr": ipAdr,
            "real_radius": real_radius,
            "comportCAM": comportCAM,
            "BoxX": BoxX,
            "BoxRX": BoxRX,
            "BoxY": BoxY,
            "BoxLEN": BoxLEN,
            "Box_i_usr": Box_i_usr,
            "cropping_val": cropping_val,
        }

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "Text Files (*.txt);;All Files (*)",
            options=QFileDialog.Options(),
        )
        if file_name:
            with open(file_name, "w", encoding="utf-8") as file:
                for key, value in saving_data.items():
                    file.write(key + " " + str(value) + "\n")

    def open_file(self):
        fname = QFileDialog.getOpenFileName(
            self, "Open file", "", "Text Files (*.txt);;All Files (*)"
        )

        if fname[0]:
            new_data = dict()
            try:
                with open(fname[0], "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split(" ")
                        if len(parts) >= 2:
                            new_data[parts[0]] = parts[1].strip()
            except FileNotFoundError:
                self.pushButton_open.setText("NOT FOUND\nBLYAT")
                return

            try:
                self.horizontalSlider_minR.setProperty("value", int(new_data["minR"]))
                self.horizontalSlider_maxR.setProperty("value", int(new_data["maxR"]))
                self.horizontalSlider_param_1.setProperty(
                    "value", int(new_data["param_1"])
                )
                self.horizontalSlider_param_2.setProperty(
                    "value", int(new_data["param_2"])
                )
                self.spinBox_scale.setProperty(
                    "value", int(float(new_data["scale"]) * 500)
                )

                global port1, ipAdr, real_radius
                port1 = int(new_data["port1"])
                ipAdr = str(new_data["ipAdr"])
                real_radius = float(new_data["real_radius"])

                self.spinBoxCOMPORT.setProperty("value", int(new_data["comportCAM"]))
                self.spinBoxX.setProperty("value", int(new_data["BoxX"]))
                self.spinBoxRX.setProperty("value", int(new_data["BoxRX"]))
                self.spinBoxY.setProperty("value", int(new_data["BoxY"]))
                self.spinBoxLEN.setProperty("value", int(new_data["BoxLEN"]))
                self.spinBox_i_usr.setProperty("value", int(new_data["Box_i_usr"]))
                self.spinBox_crop.setProperty("value", int(new_data["cropping_val"]))
            except (KeyError, ValueError):
                self.pushButton_open.setText("PARSE\nERROR")

    def usrednen(self):
        global usrednenie_flag
        if self.radioButton.isChecked():
            usrednenie_flag = True
            self.radioButton.setText("усреднение" + "\n" + "включено")
        else:
            usrednenie_flag = False
            self.radioButton.setText("усреднение" + "\n" + "отключено")
            self.factx_TEXT.setText("FactX")
            self.factY_TEXT.setText("FactY")

    def cropp(self, val_crop):
        global cropping_val
        cropping_val = int(val_crop)

    def deBox_i_averaging(self, val_i):
        global Box_i_usr
        Box_i_usr = val_i
        self.label_i_usr.setText("iуср" + "\n" + "t=" + str(Box_i_usr * 0.05)[:3])

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

    def comID(self, valu):
        global comportCAM
        comportCAM = valu

    def lineRadius(self):
        global real_radius
        real_radius = float(self.lineEdit.text())

    def linePort_2(self):
        teeeext = str(self.lineEdit_2.text()).split(":")
        global port1, ipAdr
        port1 = int(teeeext[1])
        ipAdr = str(teeeext[0])

    def valueChangesminR(self, value1):
        global minR
        minR = value1
        self.label_5.setText("minR " + str(minR))

    def valueChangesmaxR(self, value2):
        global maxR
        maxR = value2
        self.label_2.setText("maxR " + str(maxR))

    def valueChanges_param_1(self, value3):
        global param_1
        param_1 = value3
        self.label_3.setText("param1 " + str(param_1))

    def valueChanges_param_2(self, value4):
        global param_2
        param_2 = value4
        self.label_4.setText("param2 " + str(param_2))

    def valueChangesScale(self, value5):
        global scale
        scale = value5 / 500

    def init_properties(self):
        self.stream_thread = Stream_thread()
        self.Modbus_C_S = Modbus_Client_Server()

    def init_connections(self):
        self.stream_thread.change_pixmap.connect(self.image_label.setPixmap)
        self.stream_thread.update_factx.connect(self.factX.setText)
        self.stream_thread.update_facty.connect(self.factY.setText)
        self.Modbus_C_S.update_factx_text.connect(self.factx_TEXT.setText)
        self.Modbus_C_S.update_facty_text.connect(self.factY_TEXT.setText)
        self.Modbus_C_S.update_modbus_btn.connect(self.radioButtonMODBUS_START.setText)

        self.start_stop_btn.clicked.connect(self.run_stop_video_streaming)
        self.radioButtonMODBUS_START.clicked.connect(self.start_Modbus)

    @QtCore.pyqtSlot(bool)
    def start_Modbus(self):
        if self.radioButtonMODBUS_START.isChecked():
            self.Modbus_C_S.start()
            self.radioButtonMODBUS_START.setText("Started")
        else:
            self.Modbus_C_S.stop_Modbus()
            self.radioButtonMODBUS_START.setText("Stopped")

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
            icon_stop.addPixmap(
                QtGui.QPixmap(":/icons/icons/stop_video.png"),
                QtGui.QIcon.Normal,
                QtGui.QIcon.Off,
            )
            self.start_stop_btn.setIcon(icon_stop)
            self.start_stop_btn.setStyleSheet(
                "border: 2px solid red; border-radius: 7px;"
            )
        else:
            icon_run = QtGui.QIcon()
            icon_run.addPixmap(
                QtGui.QPixmap(":/icons/icons/run_video.png"),
                QtGui.QIcon.Normal,
                QtGui.QIcon.Off,
            )
            self.start_stop_btn.setIcon(icon_run)
            self.start_stop_btn.setStyleSheet(
                "border: none solid blue; border-radius: 7px;"
            )


class Modbus_Client_Server(QtCore.QThread):
    update_factx_text = QtCore.pyqtSignal(str)
    update_facty_text = QtCore.pyqtSignal(str)
    update_modbus_btn = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.thread_is_active_MODBUS = False
        self.isConnectedMod = 0
        self.PLC1 = None

    def run(self):
        oldMW3 = 0
        self.averXnumber = 0
        self.averYnumber = 0
        self.isConnectedMod = 0
        self.PLC1 = MODBUS_TCP_master()
        self.isConnectedMod = self.PLC1.Start_TCP_client(
            IP_address=ipAdr, TCP_port=port1
        )
        self.thread_is_active_MODBUS = True

        if self.isConnectedMod == 1:
            while self.thread_is_active_MODBUS:
                MW3 = self.PLC1.Read_holding_register_uint16(Register_address=BoxRX)

                if oldMW3 != MW3:
                    if usrednenie_flag:
                        averageX = []
                        averageY = []

                        with data_lock:
                            local_factx = list(factx)
                            local_facty = list(facty)

                        if local_factx and local_facty:
                            self.averXnumber = local_factx[0]
                            self.averYnumber = local_facty[0]

                            for usr in range(Box_i_usr):
                                with data_lock:
                                    local_factx = list(factx)
                                    local_facty = list(facty)

                                if (
                                    local_factx
                                    and local_facty
                                    and (
                                        self.averXnumber - 5
                                        <= local_factx[0]
                                        <= self.averXnumber + 5
                                    )
                                    and (
                                        self.averYnumber - 5
                                        <= local_facty[0]
                                        <= self.averYnumber + 5
                                    )
                                ):
                                    averageX.append(local_factx[0])
                                    self.averXnumber = sum(averageX) / len(averageX)
                                    self.update_factx_text.emit(
                                        "factX:" + str(self.averXnumber)[:5]
                                    )

                                    averageY.append(local_facty[0])
                                    self.averYnumber = sum(averageY) / len(averageY)
                                    self.update_facty_text.emit(
                                        "factY:" + str(self.averYnumber)[:5]
                                    )

                                time.sleep(0.05)

                            self.PLC1.Write_multiple_holding_register_float32(
                                Register_address=BoxX, Register_value=self.averXnumber
                            )
                            self.PLC1.Write_multiple_holding_register_float32(
                                Register_address=BoxY, Register_value=self.averYnumber
                            )
                            self.PLC1.Write_multiple_holding_register_uint16(
                                Register_address=BoxLEN, Register_value=len(local_factx)
                            )
                        else:
                            self.PLC1.Write_multiple_holding_register_uint16(
                                Register_address=BoxLEN, Register_value=0
                            )
                            self.PLC1.Write_multiple_holding_register_float32(
                                Register_address=BoxX, Register_value=0
                            )
                            self.PLC1.Write_multiple_holding_register_float32(
                                Register_address=BoxY, Register_value=0
                            )
                            self.update_factx_text.emit("factX:" + str(0))
                            self.update_facty_text.emit("factY:" + str(0))
                    else:
                        with data_lock:
                            local_factx = list(factx)
                            local_facty = list(facty)

                        if local_factx and local_facty:
                            self.PLC1.Write_multiple_holding_register_float32(
                                Register_address=BoxX, Register_value=local_factx[0]
                            )
                            self.PLC1.Write_multiple_holding_register_float32(
                                Register_address=BoxY, Register_value=local_facty[0]
                            )
                            self.PLC1.Write_multiple_holding_register_uint16(
                                Register_address=BoxLEN, Register_value=len(local_factx)
                            )
                        else:
                            self.PLC1.Write_multiple_holding_register_uint16(
                                Register_address=BoxLEN, Register_value=0
                            )
                            self.PLC1.Write_multiple_holding_register_float32(
                                Register_address=BoxX, Register_value=0
                            )
                            self.PLC1.Write_multiple_holding_register_float32(
                                Register_address=BoxY, Register_value=0
                            )

                    oldMW3 = MW3

                time.sleep(0.01)
        else:
            self.update_modbus_btn.emit(
                "нет подключения " + str(ipAdr) + ":" + str(port1)
            )

    def stop_Modbus(self):
        self.thread_is_active_MODBUS = False
        if self.isConnectedMod == 1 and self.PLC1 is not None:
            self.PLC1.Stop_TCP_client_ChutChut()
        self.wait()


class Stream_thread(QtCore.QThread):
    change_pixmap = QtCore.pyqtSignal(QtGui.QPixmap)

    update_factx = QtCore.pyqtSignal(str)
    update_facty = QtCore.pyqtSignal(str)

    def CirclesCenters(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = cv2.GaussianBlur(im, (9, 9), 2)
        rows = im.shape[0]
        circles = cv2.HoughCircles(
            im,
            cv2.HOUGH_GRADIENT,
            1,
            int(rows / 8),
            param1=param_1,
            param2=param_2,
            minRadius=minR,
            maxRadius=maxR,
        )
        center_koord = []
        current_centers = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1], i[2])

                if minR <= center[2] <= maxR:
                    current_centers.append(center)

                    if len(self.prev_center_koord) > 0:
                        distances = [
                            np.linalg.norm(np.array(center[0:2]) - np.array(prev[0:2]))
                            for prev in self.prev_center_koord
                        ]
                        closest_index = np.argmin(distances)
                        closest_distance = distances[closest_index]

                        if closest_distance < 50:
                            closest_prev = self.prev_center_koord[closest_index]
                            center = (
                                int(
                                    self.smoothing_factor * center[0]
                                    + (1 - self.smoothing_factor) * closest_prev[0]
                                ),
                                int(
                                    self.smoothing_factor * center[1]
                                    + (1 - self.smoothing_factor) * closest_prev[1]
                                ),
                                int(
                                    self.smoothing_factor * center[2]
                                    + (1 - self.smoothing_factor) * closest_prev[2]
                                ),
                            )

                    image = cv2.circle(image, center[0:-1], 1, (0, 0, 255), 3)
                    image = cv2.circle(image, center[0:-1], center[-1], (0, 250, 0), 2)
                    center_koord.append(list(map(int, center)))

            center_koord = sorted(center_koord)

            if len(center_koord) != 0:
                image = cv2.circle(
                    image,
                    (center_koord[0][0], center_koord[0][1]),
                    center_koord[0][2],
                    (255, 0, 0),
                    5,
                )

            self.prev_center_koord = current_centers
            self.prev_radius_list = [c[2] for c in current_centers]
        else:
            self.prev_center_koord = []
            self.prev_radius_list = []

        return image, center_koord

    def incr_koord_dp(self, img, l, real_radius):
        x123 = []
        y123 = []
        radius_list = []
        for i in range(len(l)):
            x123.append(l[i][0])
            y123.append(l[i][1])
            radius_list.append(l[i][2])

        h, w = img.shape[:2]
        nolx = w // 2
        noly = h // 2

        cv2.line(img, (nolx, 0), (nolx, h), (255, 0, 0), 2)
        cv2.line(img, (0, noly), (w, noly), (255, 0, 0), 2)

        for i in range(len(x123)):
            x123[i] = x123[i] - nolx
            y123[i] = y123[i] - noly

        factx_local = []
        facty_local = []
        d_real_r = []

        if real_radius != 0 and radius_list:
            for rr in range(len(radius_list)):
                d_real_r.append(real_radius / radius_list[rr])
                factx_local.append(x123[rr] * d_real_r[rr])
                facty_local.append(y123[rr] * d_real_r[rr])

        return factx_local, facty_local

    def run(self):
        self.prev_center_koord = []
        self.prev_radius_list = []
        self.smoothing_factor = 0.5
        cap = cv2.VideoCapture(comportCAM)
        self.thread_is_active = True

        try:
            while self.thread_is_active:
                ret, image = cap.read()
                if ret:
                    if (
                        (cropping_val > 1)
                        and (image.shape[0] > 2 * cropping_val)
                        and (image.shape[1] > 2 * cropping_val)
                    ):
                        image = image[
                            cropping_val : image.shape[0] - cropping_val,
                            cropping_val : image.shape[1] - cropping_val,
                        ]

                    image, center_koord = self.CirclesCenters(image)
                    new_factx, new_facty = self.incr_koord_dp(
                        image, center_koord, real_radius
                    )

                    global factx, facty

                    with data_lock:
                        factx = new_factx
                        facty = new_facty

                    if new_factx and new_facty:
                        self.update_factx.emit(str(new_factx))
                        self.update_facty.emit(str(new_facty))

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    qt_image = QtGui.QImage(
                        image_rgb.data,
                        image_rgb.shape[1],
                        image_rgb.shape[0],
                        image_rgb.strides[0],
                        QtGui.QImage.Format_RGB888,
                    ).copy()

                    pic = qt_image.scaled(
                        int(image_rgb.shape[1] * scale),
                        int(image_rgb.shape[0] * scale),
                        QtCore.Qt.KeepAspectRatio,
                    )
                    pixmap = QtGui.QPixmap.fromImage(pic)
                    self.change_pixmap.emit(pixmap)
        finally:
            cap.release()

    def stop(self):
        self.thread_is_active = False
        self.wait()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
