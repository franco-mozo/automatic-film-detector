#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:44:01 2019

@author: daiana
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np


i = 121
i = str(i).zfill(6)
file = ['./imagenes/frames_mini/' + i + '.jpg', './imagenes/' + i + 'crop.jpg']
frame = cv2.imread(file[0], 0)
crop = cv2.imread(file[1], 0)


####    Detección perforaciones     ####
########################################

# Remover ruido
frame = cv2.GaussianBlur(frame, (3,3), 0)

# Bordes Laplaciano
edges = cv2.Laplacian(frame, -1, ksize = 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
edges = cv2.dilate(edges, kernel)

plt.figure()
plt.imshow(edges, cmap = 'gray')


####    Convolución     ####
############################


def corrFFT(img, cuadro):
    
    img = img - img.mean()
    cuadro = cuadro - cuadro.mean()
    FFTimg = np.fft.fft2(img)
    FFTcuadro = np.fft.fft2(cuadro, img.shape)
    corr = abs(np.fft.ifft2(FFTimg*np.matrix.conjugate(FFTcuadro)))
    
    return corr

matching = corrFFT(frame, crop)
print('Posición frame: ', np.where(matching == np.max(matching)))