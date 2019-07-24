#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:47:59 2019

@author: daiana y fmozo
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os



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


################################# punto 4 #############################################

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
        angles[0, ang_index] = angle(x1, y1, x2, y2)
        ang_index += 1

        x1, y1 = oriented_corners[index, 0, 0], oriented_corners[index, 0, 1]
        x2, y2 = oriented_corners[index, 3, 0], oriented_corners[index, 3, 1]
        angles[0, ang_index] = angle(x1, y1, x2, y2)
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

def rotate_n_translate(src_im, corn_matrix, tmp_oriented_corners, ang_tmp):
    total_edge = 4000
    edge = 2000

    src_im_gs = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY)
    ## traslado para que coincidan la esquina 1 de la perf A de ambas imagenes
    dst = tmp_oriented_corners[0, 0, :]
    src = corn_matrix[0, 0, :]

    x = (dst[0] - src[0]).astype(int)
    y = (dst[1] - src[1]).astype(int)

    #calculo para tener espacio
    h, w = src_im.shape[0], src_im.shape[1]

    # agrando la imagen
    base = np.zeros((h+total_edge, w+total_edge))
    base[edge:h+edge,edge:w+edge] = src_im_gs[:,:]

    # traslacion
    matrix_t = np.float32([[1, 0, x], [0, 1, y]])
    translated_im = cv2.warpAffine(base, matrix_t, (h+total_edge, w+total_edge))

    # rotacion
    matrix_r = cv2.getRotationMatrix2D((dst[0]+edge, dst[1]+edge), ang_tmp, 1)
    rotated_im = cv2.warpAffine(translated_im, matrix_r, (h+total_edge, w+total_edge))

    # recorte
    if ang_tmp > 165 and (ang_tmp < 195):
        base = np.zeros((h, w))
        base[:, :] = rotated_im[edge:h+edge,edge:w+edge]
    else:
        base = np.zeros((w, h))
        base[:, :] = rotated_im[edge:w+edge,edge:h+edge]
    return base
################################# punto 5 #############################################


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


def cut_frame(I, superior, inferior, izquierdo, derecho):

    frame = I[superior + 10: inferior - 10, izquierdo + 10: derecho - 10, : ]

    return frame


#######################################################################################



def run(film_number):
    # elijo el film y leo archivos
    carpeta = film_number
    files = [img for img in os.listdir('./imagenes/' + carpeta + '/') if img.endswith('.jpg')]

    # Tratamiento del template
    template = cv2.imread('./imagenes/' + carpeta + '/' + files[0], 1)      # lectura
    found_tmp, tmp_corners = find_perf(template)        # perf encontradas en imagen y matriz
    oriented_tmp, ang_tmp = orient(template, found_tmp)         # template orientado, y angulo de giro
    tmp_oriented_corners = orient_corners(oriented_tmp, tmp_corners, ang_tmp)   # esquinas de las perf orientadas

    delta_ang = calc_rotation(tmp_oriented_corners) # calculo el angulo que falta girar
    print(ang_tmp + delta_ang*50)
    # Creacion de la carpeta destino
    if not os.path.exists('./imagenes/' + carpeta + '/aligned/'):
       os.makedirs('./imagenes/' + carpeta + '/aligned/')

    for file in files:
        # alineacion de la pelicula
        frame = cv2.imread('./imagenes/' + carpeta + '/' + file, 1)     # leo la i-esima foto

        found_perf, perf_corners = find_perf(frame)         # encuentro las perforaciones
        corn_matrix, ang = standarize_order(found_perf, perf_corners)   # estandarizo el orden de las mismas

        alined_frame = rotate_n_translate(frame, corn_matrix, tmp_oriented_corners, ang_tmp + delta_ang) # roto(antihor) y translado

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
            superiores, inferiores, derechos = det_frame(frame, tmp_oriented_corners, superiores, inferiores, derechos)
        except:
            pass

    # Corte
    superior = int(np.mean(superiores))
    inferior = int(np.mean(inferiores))
    izquierdo = int(tmp_oriented_corners[0,0,0])
    derecho = int(np.mean(derechos))

    for file in files:

        frame = cv2.imread('./imagenes/' + carpeta + '/aligned/' + file, 1)

        frame = cut_frame(frame, superior, inferior, izquierdo, derecho)

        if os.path.exists('./imagenes/' + carpeta + '/movie/' + file):
            os.remove('./imagenes/' + carpeta + '/movie/' + file)

        cv2.imwrite('./imagenes/' + carpeta + '/movie/' + file, frame)

    return

run('film_4')
