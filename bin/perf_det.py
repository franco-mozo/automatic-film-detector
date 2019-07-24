#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:27:02 2019
@author: daiana
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np


def OrientDet(frame):

    # Remover ruido
    frame = cv2.GaussianBlur(frame, (3,3), 0)

    # Bordes Laplaciano
    edges = cv2.Laplacian(frame, -1, ksize = 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    edges = cv2.dilate(edges, kernel)

#    plt.figure()
#    plt.imshow(edges, cmap = 'gray')
#    plt.title('Laplaciano')

    ### Binarización

    binaria = cv2.cvtColor(edges,cv2.COLOR_BGR2GRAY)  # Debe estar en blanco y negro para poder usar threshold
    ret, binaria = cv2.threshold(binaria,80,255,cv2.THRESH_BINARY)

#    plt.figure()
#    plt.imshow(binaria, cmap = 'gray')
#    plt.title('Laplaciano Binarizado')

    ### Detección de líneas aplicado al Laplaciano

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 150 # minimum number of pixels making up a line
    max_line_gap = 0  # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(binaria, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    return line_image

film_1 = ['./imagenes/film_1/000121.jpg']
film_2 = ['./imagenes/film_2/000819.jpg']
film_3 = ['./imagenes/film_3/000441.jpg']
film_4 = ['./imagenes/film_4/000160.jpg']
film_5 = ['./imagenes/film_5/000346.jpg']

plt.figure()
plt.subplot(1,5,1)
frame = cv2.imread(film_1[0], 1)
plt.imshow(OrientDet(frame))
plt.title('film_1')

plt.subplot(1,5,2)
frame = cv2.imread(film_2[0], 1)
plt.imshow(OrientDet(frame))
plt.title('film_2')

plt.subplot(1, 5, 3)
frame = cv2.imread(film_3[0], 1)
plt.imshow(OrientDet(frame))
plt.title('film_3')

plt.subplot(1,5,4)
frame = cv2.imread(film_4[0], 1)
plt.imshow(OrientDet(frame))
plt.title('film_4')

plt.subplot(1,5,5)
frame = cv2.imread(film_5[0], 1)
plt.imshow(OrientDet(frame))
plt.title('film_5')
