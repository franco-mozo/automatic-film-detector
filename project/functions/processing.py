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

#######################################################################################


def orient(I, found_perf):
    vert = found_perf[:,1100:]      # franja con posibles perforaciones verticales
    horiz = found_perf[650:,:]     # franja con posibles perforaciones horizontales

    # perforaciones están en la franja horizontal o vertical?
    if len(vert[vert == 255]) > len(horiz[horiz == 255]):   # si están en la vertical, rotar 180 grados
        M = cv2.getRotationMatrix2D((I.shape[1]/2, I.shape[0]/2), 180, 1)
        I = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]))
        ang = 180

    else:       # si están en la horizontal, rotar 90 grados
        M = cv2.getRotationMatrix2D((I.shape[1]/3, I.shape[0]/2), 270, 1)
        I = cv2.warpAffine(I, M, (I.shape[0], I.shape[1]))
        ang = 270

    return cv2.cvtColor(I, cv2.COLOR_BGR2RGB), ang
#######################################################################################

def orient_corners(oriented_frame, perf_corners, ang):
    h ,w = oriented_frame.shape[0], oriented_frame.shape[1]
    oriented_corners = perf_corners.copy()
    index = 0

    aux_h = np.ones((4,))*h
    aux_w = np.ones((4,))*w
    if ang == 180:
        while perf_corners[index, 0, 0] != -1:
            oriented_corners[index, :, 0] = aux_w - perf_corners[index, :, 0]
            oriented_corners[index, :, 1] = aux_h - perf_corners[index, :, 1]
            index += 1

    elif ang == 270:
        aux_matrix = perf_corners.copy()
        while perf_corners[index, 0, 0] != -1:
            aux_matrix[index, :, 0] = aux_w - aux_matrix[index, :, 1]
            aux_matrix[index, :, 1] = perf_corners[index, :, 0]

            # para que guarde desde la esquina superior derecha de las perforaciones
            aux_matrix[index] = np.roll(aux_matrix[index],-2)
            index += 1
        # queda en index el # de perforaciones
        for i in range(0,index):
            oriented_corners[i, :, :] = aux_matrix[index-1-i, :, :]

    return oriented_corners
#######################################################################################


def standarize_order(found_perf, perf_corners):
    corn_matrix = perf_corners.copy()

    # calculo el caso
    vert = found_perf[:,1100:]
    horiz = found_perf[650:,:]
    if len(vert[vert == 255]) > len(horiz[horiz == 255]):
        ang = 180
    else:
        ang = 270

    # si es el caso 270 grados, reordeno la matriz
    if ang == 270:
        # reordeno orientacion de esquinas
        index = 0
        while perf_corners[index, 0, 0] != -1:
            #-----
            box = perf_corners[index, :, :].copy()
            aux = perf_corners[index, :, :].copy()  # aux = box ordenado
            aux[0,:],aux[1,:],aux[2,:],aux[3,:], = box[1,:],box[2,:],box[3,:],box[0,:]
            corn_matrix[index,:,:] = aux[:,:]
            index += 1
            #-----
        # reordeno orden de las perforaciones
        # queda en index el # de perforaciones
        aux_matrix = corn_matrix.copy()
        for i in range(0,index):
            corn_matrix[i, :, :] = aux_matrix[index-1-i, :, :]

    return corn_matrix, ang
#######################################################################################


def angle(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    else:
        return np.arctan((y2 - y1) / (x2 - x1))
#######################################################################################


def rotate_n_move(src_im, corn_matrix, tmp_oriented_corners, ang_tmp):
    total_edge = 4000
    edge = 2000

    ## traslado para que coincidan la esquina 1 de la perf A de ambas imagenes
    dst = tmp_oriented_corners[0, 0, :]
    src = corn_matrix[0, 0, :]

    x = (dst[0] - src[0]).astype(int)
    y = (dst[1] - src[1]).astype(int)

    #calculo para tener espacio
    h, w, l = src_im.shape

    # agrando la imagen
    base = np.zeros((h+total_edge, w+total_edge, l), dtype = np.uint8)
    base[edge:h+edge,edge:w+edge,:] = src_im[:,:,:]

    # traslacion
    matrix_t = np.float32([[1, 0, x], [0, 1, y]])
    translated_im = cv2.warpAffine(base, matrix_t, (h+total_edge, w+total_edge))

    # rotacion
    matrix_r = cv2.getRotationMatrix2D((dst[0]+edge, dst[1]+edge), ang_tmp, 1)
    rotated_im = cv2.warpAffine(translated_im, matrix_r, (h+total_edge, w+total_edge))

    # recorte
    if ang_tmp > 165 and (ang_tmp < 195):
        base = np.zeros((h, w, l))
        base[:, :, :] = rotated_im[edge:h+edge,edge:w+edge,:]
    else:
        base = np.zeros((w, h, l))
        base[:, :, :] = rotated_im[edge:w+edge,edge:h+edge,:]
    return base.astype(np.uint8)
#######################################################################################


def cut_frame(I, superior, inferior, izquierdo, derecho):

    frame = I[superior + 10: inferior - 10, izquierdo + 10: derecho - 10, : ]

    return frame
#######################################################################################
