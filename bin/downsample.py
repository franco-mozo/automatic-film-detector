#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 19:33:24 2019

@author: daiana
"""

################
### Asegurarse de que la carpeta de imágenes originales y una carpeta de nombre frames_mini, estén
### en la misma ruta que este .py antes de ejecutar.
################

import cv2


for i in range(17, 452):

    i = str(i).zfill(6)
    frame = cv2.imread('./imagenes/film_4/44/SE044_' + i + '.jpg', 1)
    newframe = cv2.resize(frame, (int(frame.shape[1]/4),int(frame.shape[0]/4)))
    
    cv2.imwrite('./imagenes/film_4/' + i + '.jpg', newframe)