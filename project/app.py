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
import os
from os.path import isfile, join
#######################################################################################
from project.functions import detection, processing, inputf, output


def run():
    # elijo el film y leo archivos
    carpeta, fps = inputf.user_requeriments()

    files = [img for img in os.listdir('./imagenes/' + carpeta + '/') if img.endswith('.jpg')]

    # Tratamiento del template
    template = cv2.imread('./imagenes/' + carpeta + '/' + files[0], 1)      # lectura
    found_tmp, tmp_corners = detection.find_perf(template)        # perf encontradas en imagen y matriz
    oriented_tmp, ang_tmp = processing.orient(template, found_tmp)         # template orientado, y angulo de giro
    tmp_oriented_corners = processing.orient_corners(oriented_tmp, tmp_corners, ang_tmp)   # esquinas de las perf orientadas

    delta_ang = detection.calc_rotation(tmp_oriented_corners) # calculo el angulo que falta girar

    # Creacion de la carpeta destino
    if not os.path.exists('./imagenes/' + carpeta + '/aligned/'):
       os.makedirs('./imagenes/' + carpeta + '/aligned/')

    for file in files:
        # alineacion de la pelicula
        frame = cv2.imread('./imagenes/' + carpeta + '/' + file, 1)     # leo la i-esima foto

        found_perf, perf_corners = detection.find_perf(frame)         # encuentro las perforaciones
        corn_matrix, ang = processing.standarize_order(found_perf, perf_corners)   # estandarizo el orden de las mismas

        alined_frame = processing.rotate_n_move(frame, corn_matrix, tmp_oriented_corners, ang_tmp + delta_ang) # roto(antihor) y translado

        if os.path.exists('./imagenes/' + carpeta + '/aligned/' + file):
            os.remove('./imagenes/' + carpeta + '/aligned/' + file)

        cv2.imwrite('./imagenes/' + carpeta + '/aligned/' + file, alined_frame)   # escribo a memoria

    # Creacion de la carpeta destino
    if not os.path.exists('./imagenes/' + carpeta + '/movie/'):
       os.makedirs('./imagenes/' + carpeta + '/movie/')

    # Determinación de límites de corte
    superiores = []
    inferiores = []
    derechos = []

    for file in files:

        frame = cv2.imread('./imagenes/' + carpeta + '/aligned/' + file, 1)

        try:
            superiores, inferiores, derechos = detection.det_frame(frame, tmp_oriented_corners, superiores, inferiores, derechos)
        except:
            pass

    # si se encontraron bordes, se trabaja con eso, sino se deben ingresar a mano
    if (superiores == []) or (inferiores == []) or (derechos == []):

        frame = cv2.imread('./imagenes/' + carpeta + '/aligned/' + files[0], 1)
        superior, inferior, izquierdo, derecho = detection.manual_input(frame)
    else:

        superior = int(np.mean(superiores))
        inferior = int(np.mean(inferiores))
        izquierdo = int(tmp_oriented_corners[0,0,0])
        derecho = int(np.mean(derechos))

    for file in files:

        frame = cv2.imread('./imagenes/' + carpeta + '/aligned/' + file, 1)

        frame = processing.cut_frame(frame, superior, inferior, izquierdo, derecho)

        if os.path.exists('./imagenes/' + carpeta + '/movie/' + file):
            os.remove('./imagenes/' + carpeta + '/movie/' + file)

        cv2.imwrite('./imagenes/' + carpeta + '/movie/' + file, frame)

    output.video_write(carpeta, fps)
    print('Done :::: SUCCESS ::::')
    return
