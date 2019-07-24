#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 19 Jul 2019

@author: daiana y fmozo
"""
################################### Imports ###########################################
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import numpy as np

from project.functions import processing
#######################################################################################


def find_perf(I):
    ##############################################################
    ## Devuelve imagen con perf encontradas y matriz (cubo) con ##
    ## esquinas de las perf orden (II,SI,SD,ID)                 ##
    ##############################################################

    ret, J = cv2.threshold(I,200,255,cv2.THRESH_TOZERO)

    kernel = np.ones((5,5),np.uint8)
    J = cv2.erode(J,kernel,iterations = 2)
    kernel = np.ones((4,4),np.uint8)
    J = cv2.dilate(J,kernel, iterations = 2)

    ret, J = cv2.threshold(J,200,255,cv2.THRESH_BINARY)

    found_perf = np.zeros(J.shape)
    J = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
    contours,_ = cv2.findContours(J, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    perf_corners = np.zeros((40,4,2)).astype(int)
    index = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        # box = vertices del rect sent antihorario arranca
        # esquina superior derecha
        box = np.int0(cv2.boxPoints(rect))

        # Calculo datos de interes de los rectangulos
        x_min = np.amin(box, axis=0)[0]
        x_max = np.amax(box, axis=0)[0]

        y_min = np.amin(box[:,1])
        y_max = np.amax(box[:,1])

        #filtro por area y long de los lados
        h,w = y_max-y_min, x_max-x_min
        area = h*w
        area_range_ok = (area >3200) and (area<105000)
        dim_proportion_ok = np.abs(h-w) < 30
        if area_range_ok and dim_proportion_ok :
            cv2.drawContours(found_perf,[box],0,(255),3)

            # reordeno los vertices y los guardo
            x1 = box[0,0]
            x3 = box[2,0]
            aux = box.copy()  # aux = box ordenado
            if x1 >= x3 : #hay que rotar
                aux[0,:],aux[1,:],aux[2,:],aux[3,:], = box[1,:],box[2,:],box[3,:],box[0,:]
            perf_corners[index,:,:] = aux[:,:]
            index += 1

    perf_corners[index,:,:] = np.ones((4,2)).astype(int)*(-1)
    return found_perf, perf_corners
#######################################################################################


def calc_rotation(oriented_corners):
    ###############################################################
    # calcula mediante promedio de pendientes, el angulo a rotar ##
    ###############################################################
    angles = np.zeros((1, 30))
    ang_index = 0
    index = 0
    while oriented_corners[index, 0, 0] != -1:
        x1, y1 = oriented_corners[index, 1, 0], oriented_corners[index, 1, 1]
        x2, y2 = oriented_corners[index, 2, 0], oriented_corners[index, 2, 1]
        angles[0, ang_index] = processing.angle(x1, y1, x2, y2)
        ang_index += 1

        x1, y1 = oriented_corners[index, 0, 0], oriented_corners[index, 0, 1]
        x2, y2 = oriented_corners[index, 3, 0], oriented_corners[index, 3, 1]
        angles[0, ang_index] = processing.angle(x1, y1, x2, y2)
        ang_index += 1
        index += 1
    angles[0, ang_index] = -1

    # promedio todos los angulos calculados
    i = 0
    ang_mean = 0
    while angles[0, i] != -1:
        ang_mean += angles[0, i]
        i += 1
    ang_mean = ang_mean / i
    return ang_mean
#######################################################################################


def borde(franja):

    blur = cv2.GaussianBlur(franja,(5,5),0)

    edges = cv2.Laplacian(blur, -1, ksize = 5)

    _, binaria = cv2.threshold(edges,80,255,cv2.THRESH_BINARY)
    binaria = cv2.cvtColor(binaria, cv2.COLOR_RGB2GRAY)

    ### Detección de líneas aplicado al Laplaciano

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Houg, cmap = 'gray'h grid cell)
    min_line_length = 160 # minimum number of pixels making up a line
    max_line_gap = 0  # maximum gap in pixels between connectable line segments
    line_image = np.copy(binaria) * 0  # creating a blank to draw lines on


    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(binaria, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),1)

    return line_image, lines
#######################################################################################


def det_frame(I, x, superior, inferior, derecho):

    sup = I[int(x[0,0,1]): int(x[0,1,1]), int(x[0,0,0]):, :]
    img1, bordesup = borde(sup)
    superior.append(int(x[0,0,1]) + np.max(bordesup[:,0,1]))

    inf = I[int(x[1,0,1]): int(x[1,1,1]), int(x[1,0,0]):, :]
    img2, bordeinf = borde(inf)
    inferior.append(int(x[1,0,1]) + np.min(bordeinf[:,0,1]))

    # Tomo un margen de +20 y -20 arriba y abajo para evitar posibles errores
    inter = I[int(x[0,0,1]) + bordesup[0,0,1] + 20: int(x[1,0,1]) + bordeinf[0,0,1] - 20, int(x[0,0,0]) + 400: , :]
    img3, bordeder = borde(inter)
    derecho.append(int(x[0,0,0]) + 400 + np.min(bordeder[:,0,0]))

    return superior, inferior, derecho
#######################################################################################


def manual_input(I):

    plt.figure()
    plt.imshow(I)
    plt.title('Marque esquinas superior izquierda e inferior derecha del frame:')
    p = plt.ginput(2)

    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return int(p[0][1]), int(p[1][1]), int(p[0][0]), int(p[1][0])
#######################################################################################
