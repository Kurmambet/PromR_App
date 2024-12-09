import cv2
import numpy as np

import imutils
import os
from MODBUS_TCP_CLIENT import *
from concurrent.futures import ThreadPoolExecutor

# import threading
# from threading import Thread



def resized(image, final_wide=800): #функция пропорционального изменения размера
    # Подготовим новые размеры
    r = float(final_wide) / image.shape[1]
    dim = (final_wide, int(image.shape[0] * r))
    # уменьшаем изображение до подготовленных размеров
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def CirclesCenters(img, m1, m2, minR, maxR, scale):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = im.shape[0]

    circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=m1, param2=m2,
                               minRadius=minR, maxRadius=maxR)
    center_koord = []
    radius_list = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            center_koord.append(list(map(int, center)))

            # circle center
            img = cv2.circle(img, center, 1, (0, 0, 255), 3)
            # circle outline
            radius = i[2]
            img = cv2.circle(img, center, radius, (0, 255, 0), 3)
            radius_list.append(int(i[2]))



        print('radius_list!!!!!',radius_list)
        print('center_koord!!!!!', center_koord)

    #cv2.imshow('from camera', resized(img, scale))

    return center_koord, radius_list


def incr_koord_dp(img, l, radius_list, real_radius,  scale, is_setting):

    # пикселей на мм по оси х и у
    #расчет абсолютных координат
    x123 = []
    y123 = []
    for i in range(len(l)):
        x123.append(l[i][0])
        y123.append(l[i][1])

    if is_setting == 1:
        print('absolutx', x123)
        print('absoluty', y123)
        image = (np.zeros((img.shape[0],img.shape[1],3),np.uint8))

    # for j in range(len(l)):
    #     image = cv2.circle(img,(int(x123[j]), int(y123[j])),1,(0, 0, 255), 10)
    # cv2.imshow('image111111111111111111', resized(image, scale))

    #вычисление новых коорд относительно центра кадра

    (h, w) = img.shape[:2]

    nolx = w // 2
    noly = h // 2

    for i in range(len(x123)):
        x123[i] = x123[i] - nolx
        y123[i] = y123[i] - noly




    # print('размеры картинки',w,h)


    image2 = (np.zeros((h, w, 3), np.uint8))

    for j in range(len(x123)):
        image2 = cv2.circle(image2,(int(x123[j] + nolx), int(y123[j] + noly)),1,(0, 0, 255), 5)

        cv2.line(img, (x123[j] + nolx + radius_list[j],   y123[j] + noly), (int(x123[j] + nolx), int(y123[j] + noly)), (255, 255, 255), 3)
        #cv2.line(image2, (radius_list[0], radius_list[0]), (nolx, noly), (0, 255, 0), thickness=5)
        #
        # cv2.line(img, (nolx + radius_list[j], noly + radius_list[j]), (nolx, noly), (255, 255, 255), thickness=3)
        # cv2.line(image2, (radius_list[j] + nolx, radius_list[j] + noly), (nolx, noly), (0, 255, 0), thickness=5)

    # центр коорд на исходном
    cv2.circle(img, (nolx, noly), 5, (255, 0, 0), -1)



    # центр коорд на отфильтрованном
    image2 = cv2.circle(image2, (nolx, noly), 20, (255, 0, 0), -1)



    if is_setting == 1:
        print('середина кадра',nolx,noly)
        print('centerx', x123)
        print('centery', y123)
        cv2.imshow('image2', resized(image2, scale))

    cv2.imshow('from camera', resized(img, scale))


    # cv2.imshow('from camera', resized(img, scale))

    factx = []
    facty = []
    d_real_r = []
    mm_nolx = mm_noly = 0

    rast_o_dot_p = []
    rast_o_dot_mm = []

    if  real_radius != 0 and radius_list:



        for rr in range(len(radius_list)):
            d_real_r.append(real_radius/radius_list[rr])

            factx.append(x123[rr] * d_real_r[rr])
            facty.append(y123[rr] * d_real_r[rr])


        for itr in range(len(x123)):
            mm_nolx = ((sum(d_real_r)/len(d_real_r)) * nolx)
            mm_noly = ((sum(d_real_r)/len(d_real_r)) * noly)

            rast_o_dot_p.append((((x123[itr]) ** 2) + ((y123[itr]) ** 2)) ** 0.5)

            rast_o_dot_mm.append((((factx[itr]) ** 2) + ((facty[itr]) ** 2)) ** 0.5)



        if is_setting == 1:

            print('Коэффициент отношения мм к пикселям', d_real_r)
            print('x координаты в мм отн. центра кадра', factx)
            print('y координаты в мм отн. центра кадра', facty)

            print('координаты центра кадра в мм       ', mm_nolx, mm_noly)
            print('расстояние до центров в пикселях   ', rast_o_dot_p)
            print('расстояние до центров в мм         ', rast_o_dot_mm)

    else:
        facty = []
        factx = []


    return factx, facty

if __name__ == '__main__':
    def nothing(*arg):
        pass




def Settings():
    def create_Settings():
        cv2.namedWindow("settings")
        cv2.resizeWindow("settings", 550, 200)
        cv2.createTrackbar('scale', 'settings', 800, 1500, nothing)
        cv2.createTrackbar('circle_param1', 'settings', 100, 255, nothing)
        cv2.createTrackbar('circle_param2', 'settings', 28, 255, nothing)
        cv2.createTrackbar('min_R', 'settings', 50, 255, nothing)
        cv2.createTrackbar('max_R', 'settings', 170, 455, nothing)



    def Verse_Settings():

        real_radius = int(input('введите радиус данной заготовки в мм'))

        # cap = cv2.VideoCapture(0)
        # if not cap.isOpened():
        #     print("Cannot open camera")
        #     exit()
        while True:
            # ret, img = cap.read()
            #
            # if not ret:
            #     print("Can't receive frame (stream end?). Exiting ...")
            #     break


            img = cv2.imread('D:/programirovanie/kolab_zrenie/testcv/KVADRKRUG2.jpg')
            scale = cv2.getTrackbarPos('scale', 'settings')
            m1 = cv2.getTrackbarPos('circle_param1', 'settings')
            m2 = cv2.getTrackbarPos('circle_param2', 'settings')
            minR = cv2.getTrackbarPos('min_R', 'settings')
            maxR = cv2.getTrackbarPos('max_R', 'settings')

            if m1 > 30 and m2 > 20 and minR > 5 and maxR > 2:
                with ThreadPoolExecutor() as executor2:
                    future2 = executor2.submit(CirclesCenters, img, m1, m2, minR, maxR, scale)
                    center_koord, radius_list = future2.result()
            else:
                center_koord = []
                radius_list = []


            factx, facty = incr_koord_dp(img, center_koord, radius_list, real_radius, scale,1)


            ch = cv2.waitKey(5)
            if ch == 27:

                print('param in func ', m1, m2, minR, maxR, real_radius, scale)

                actual_parameters = [m1, m2, minR, maxR, real_radius, scale]



                if len(center_koord) > 0:
                    incr_koord_dp(img, center_koord, radius_list, real_radius, scale,1)
                else:
                    print('кругов на кадре нет')

                return actual_parameters

                cv2.destroyAllWindows()


    create_Settings()
    actual_parameter = Verse_Settings()
    return actual_parameter

actual_parameter_itog = Settings()
cv2.destroyAllWindows()
print('OUTFUNC',actual_parameter_itog)





def Unit_test(actual_parameter_itog):

    m1, m2, minR, maxR, real_radius, scale = actual_parameter_itog[0], actual_parameter_itog[1],actual_parameter_itog[2],actual_parameter_itog[3],actual_parameter_itog[4],actual_parameter_itog[5]
    print('UNITTTTTEST', m1, m2, minR, maxR, real_radius, scale)
    xi = 0
    yi = 0
    oldMW3 = 0

    PLC1 = MODBUS_TCP_master()
    PLC1.Start_TCP_client(IP_address='127.0.0.1')




    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        #img111 = cv2.imread('D:/programirovanie/kolab_zrenie/testcv/KVADRKRUG2.jpg')

        ret, img111 = cap.read()

        #if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # cv2.imshow('real', resized(img111,800))

        if cv2.waitKey(1) == ord('q'):
            break

        MW3 = PLC1.Read_holding_register_uint16(Register_address=3)

        with ThreadPoolExecutor() as executor2:
            future2 = executor2.submit(CirclesCenters, img111, m1, m2, minR, maxR, scale)
            center_koord, radius_list = future2.result()

        newmmx, newmmy = incr_koord_dp(img111, center_koord, radius_list, real_radius, scale,0)


        if oldMW3 != MW3:

            if newmmx:
                if MW3 in range(len(newmmx)):
                    PLC1.Write_multiple_holding_register_float32(Register_address=1, Register_value=newmmx[MW3])
                    PLC1.Write_multiple_holding_register_float32(Register_address=4, Register_value=newmmy[MW3])
                    oldMW3 = MW3
            else: continue

    cap.release()
    cv2.destroyAllWindows()
    PLC1.Stop_TCP_client()



Unit_test(actual_parameter_itog)
