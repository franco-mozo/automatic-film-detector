#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:47:59 2019

@author: daiana
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os



def find_perf(I):
	##############################################################
	## Devuelve imagen con perf encontradas y matriz (cubo) con ##
	## esquinas de las perf orden (II,SI,SD,ID) 				##
	##############################################################

	ret, J = cv2.threshold(I,200,255,cv2.THRESH_TOZERO)

	kernel = np.ones((5,5),np.uint8)
	J = cv2.erode(J,kernel,iterations = 2)
	kernel = np.ones((4,4),np.uint8)
	J = cv2.dilate(J,kernel, iterations = 2)

	ret, J = cv2.threshold(J,200,255,cv2.THRESH_BINARY)
	
	found_perf = np.zeros(J.shape)
	J = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
	_,contours,_ = cv2.findContours(J, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

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
		area_range_ok = (area >2500) and (area<105000)
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


def corners(I, perf_corners):

    #for i in len(perf_corners):
    i = 0
    while perf_corners[i,0,0] != -1:
        for corner in perf_corners[i]:
            	x, y = corner.ravel()
            	cv2.circle(I, (x,y), 10, 150, -1)
        i += 1
            
    return I
    

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
        while perf_corners[index, 0, 0] != -1:
            oriented_corners[index, :, 0] = aux_w - perf_corners[index, :, 1]
            oriented_corners[index, :, 1] = perf_corners[index, :, 0]
            
            # para que guarde desde la esquina superior derecha de las perforaciones
            oriented_corners[index] = np.roll(oriented_corners[index],-2)
            index += 1    

    return oriented_corners


def angle(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    else:
        return np.arctan((y2 - y1) / (x2 - x1))


def calc_rotation(oriented_corners):
    ###############################################################
    # calcula mediante promedio de pendientes, el angulo a rotar ##
    ###############################################################
    angles = np.zeros((1, 40))
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


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def borde(franja):

#    franja = cv2.cvtColor(franja, cv2.COLOR_BGR2GRAY)
#    franja = cv2.equalizeHist(franja)
#    franja = cv2.cvtColor(franja, cv2.COLOR_GRAY2BGR)
    
    edges = cv2.Laplacian(franja, -1, ksize = 5)
    
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


def det_frame(I, x, ang):
    
    # descarto 0 y -1
    x = x[x[:,0,0] != 0]
    x = x[x[:,0,0] != -1]
    
    if ang == 270:
        # para que las dos primeras perforaciones se lean de arriba a abajo
        x = np.roll(x,2,0) 
        x[0:2] = np.roll(x[0:2], 1, 0)

    sup = I[int(x[0,0,1]): int(x[0,1,1]), int(x[0,0,0]):, :]
    img1, bordesup = borde(sup)
    
    inf = I[int(x[1,0,1]): int(x[1,1,1]), int(x[1,0,0]):, :]
    img2, bordeinf = borde(inf)
    
    # Tomo un margen de +20 y -20 arriba y abajo para evitar posibles errores
    inter = I[int(x[0,0,1]) + bordesup[0,0,1] + 20: int(x[1,0,1]) + bordeinf[0,0,1] - 20, int(x[0,0,0]) + 400: , :]
    img3, bordeder = borde(inter)
    
    cuadro = I[int(x[0,0,1]) + np.max(bordesup[:,0,1]): int(x[1,0,1]) + np.min(bordeinf[0,0,1]), int(x[0,0,0]): int(x[0,0,0])  + 400 + np.min(bordeder[:,0,0]) , :]
    
    return cuadro



carpeta = 'film_2'

files = [img for img in os.listdir('./imagenes/' + carpeta + '/') if img.endswith('.jpg')]

if not os.path.exists('./imagenes/' + carpeta + '/movie/'):
   os.makedirs('./imagenes/' + carpeta + '/movie/') 

for file in files:

    frame = cv2.imread('./imagenes/' + carpeta + '/' + file, 1)
    perf, esquinas = find_perf(frame)
    frame, ang = orient(frame, perf)
    esquinas = orient_corners(frame, esquinas, ang)
    
    try:
        frame = det_frame(frame, esquinas, ang)
    except:
        pass
    
    if os.path.exists('./imagenes/' + carpeta + '/movie/' + file):
        os.remove('./imagenes/' + carpeta + '/movie/' + file)
        
    cv2.imwrite('./imagenes/' + carpeta + '/movie/' + file, frame)
    