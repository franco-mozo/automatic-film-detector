#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:45:50 2019

@author: daiana
"""

import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import numpy as np
import os

film_1 = ['./imagenes/film_1/000121.jpg']
film_2 = ['./imagenes/film_2/000819.jpg']
film_3 = ['./imagenes/film_3/000441.jpg']
film_4 = ['./imagenes/film_4/000160.jpg']
film_5 = ['./imagenes/film_5/000346.jpg']


'''files = os.listdir('./imagenes/film_2/')'''

#def fill_holes(gray):
#    edges = cv2.Canny(gray, 80, 640)
##    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
##    edges = cv2.dilate(edges, kernel)
#    
#    edges = cv2.dilate(edges, None, iterations=8)
#    #edges = cv2.erode(edges, None, iterations=8)
#    
#    fill_holes = ndi.binary_fill_holes(edges)
#    fill_holes = fill_holes.astype(np.uint8)
#    
#    _,contours,_ = cv2.findContours(fill_holes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#
#    perf = np.zeros(fill_holes.shape)
#    
#    for cnt in contours:
#        	rect = cv2.minAreaRect(cnt)
#        	# box = vertices del rect sent antihorario arranca 
#        	# esquina superior derecha
#        	box = np.int0(cv2.boxPoints(rect)) 
#        	#filtro por area y long de los lados
#        	h,w = box[3,0]-box[0,0], box[3,1]-box[2,1]
#        	#print(h) 
#        	if (int(h*w) > 500) and (h>30):
#        		cv2.drawContours(perf,[box],0,(255),3)
#  
#    return perf


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


#def corners(I):
#    
#    I = np.float32(I)
#    
#    corners = cv2.goodFeaturesToTrack(I,40,0.01,50)
#    corners = np.int0(corners)
#
#    for corner in corners:
#        	x, y = corner.ravel()
#        	cv2.circle(I, (x,y), 10, 150, -1)
#            
#    return I


def corners(I, perf_corners):

    #for i in len(perf_corners):
    i = 0
    while perf_corners[i,0,0] != -1:
        for corner in perf_corners[i]:
            	x, y = corner.ravel()
            	cv2.circle(I, (x,y), 10, 150, -1)
        i += 1
            
    return I


def orient(I, perf):
    
    vert = perf[:,1100:]      # franja con posibles perforaciones verticales
    horiz = perf[650:,:]     # franja con posibles perforaciones horizontales
    
    # perforaciones están en la franja horizontal o vertical?
    if len(vert[vert == 255]) > len(horiz[horiz == 255]):   # si están en la vertical, rotar 180 grados
        M = cv2.getRotationMatrix2D((I.shape[1]/2, I.shape[0]/2), 180, 1)
        I = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]))
        I = I.astype(np.uint8)
    
    else:       # si están en la horizontal, rotar 90 grados
        M = cv2.getRotationMatrix2D((I.shape[1]/2, I.shape[0]/2), 270, 1)
        I = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]))
        I = I.astype(np.uint8)
        
    return cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    

#def orient(I, found_perf, perf_corners):
#    
#    vert = found_perf[:,1100:]      # franja con posibles perforaciones verticales
#    horiz = found_perf[650:,:]     # franja con posibles perforaciones horizontales
#    
#    perf_corners = perf_corners.astype(np.float32)
#    
#    # perforaciones están en la franja horizontal o vertical?
#    if len(vert[vert == 255]) > len(horiz[horiz == 255]):   # si están en la vertical, rotar 180 grados
#        M = cv2.getRotationMatrix2D((I.shape[1]/2, I.shape[0]/2), 180, 1)
#        I = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]))
#        
#        # correción de índices en la matriz de vértices
#        i = 0
#        while perf_corners[i,0,0] != -1:
#            perf_corners[i,:,0] = I.shape[0] - perf_corners[i,:,0]
#            perf_corners[i,:,1] = I.shape[1] - perf_corners[i,:,1]
#            i += 1
#            
#    else:       # si están en la horizontal, rotar 90 grados
#        M = cv2.getRotationMatrix2D((I.shape[1]/2, I.shape[0]/2), 270, 1)
#        I = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]))
#
#        # correción de índices en la matriz de vértices
#        i = 0
#        while perf_corners[i,0,0] != -1:
#            aux = perf_corners[i,:,0].copy()
#            perf_corners[i,:,0] = perf_corners[i,:,1]
#            perf_corners[i,:,1] = aux
#            i += 1
#   
#    return cv2.cvtColor(I, cv2.COLOR_BGR2RGB), perf_corners


def angle(x1, y1, x2, y2):
	if x1 == x2:
		return 0
	else:
		return np.arctan((y2-y1)/(x2-x1))
    
    
def calc_rotation(perf_corners):
	################################################################
	## calcula mediante promedio de pendientes, el angulo a rotar ##
	################################################################
	angles = np.zeros((1,40))
	ang_index = 0 
	index = 0
	while perf_corners[index,0,0] != -1:
		x1, y1 = perf_corners[index,1,0],perf_corners[index,1,1]
		x2, y2 = perf_corners[index,2,0],perf_corners[index,2,1]
		angles[0,ang_index] = angle(x1, y1, x2, y2)
		ang_index += 1

		x1, y1 = perf_corners[index,0,0],perf_corners[index,0,1]
		x2, y2 = perf_corners[index,3,0],perf_corners[index,3,1]
		angles[0,ang_index] = angle(x1, y1, x2, y2)
		ang_index += 1
		index+=1
	angles[0,ang_index] = -1

	# promedio todos los angulos calculados
	i = 0
	ang_mean = 0
	while angles[0,i] != -1:
		ang_mean += angles[0,i]
		i += 1
	ang_mean = ang_mean/i
	return ang_mean


def rotateImage(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result


def borde(franja):
    
    edges = cv2.Laplacian(franja, -1, ksize = 5)
    
    _, binaria = cv2.threshold(edges,80,255,cv2.THRESH_BINARY)
    binaria = cv2.cvtColor(binaria, cv2.COLOR_RGB2GRAY)
    
    ### Detección de líneas aplicado al Laplaciano

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Houg, cmap = 'gray'h grid cell)
    min_line_length = 150 # minimum number of pixels making up a line
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


#def det_frame(I, x):
#    
#    sup = I[int(x[0,1]): int(x[1,1]), int(x[0,0]):, :]
#    img1, bordesup = borde(sup)
#    
#    inf = I[int(x[2,1]): int(x[3,1]), int(x[0,0]):, :]
#    img2, bordeinf = borde(inf)
#    
#    # Tomo un margen de +20 y -20 arriba y abajo para evitar posibles errores
#    inter = I[int(x[0,1]) + bordesup[0,0,1] + 20: int(x[2,1]) + bordeinf[0,0,1]-20, int(x[0,0]): , :]
#    img3, bordeder = borde(inter)
#    
#    cuadro = I[int(x[0,1]) + np.max(bordesup[:,0,1]): int(x[2,1]) + np.min(bordeinf[0,0,1]), int(x[0,0]): int(x[0,0]) + np.min(bordeder[:,0,0]) , :]
#    
#    return cuadro
    

#def det_frame(I, x):
#    
#    # Asumiendo que la primer perforación que llega es la superior
#    sup = I[int(x[0,2,1]): int(x[0,3,1]), int(x[0,2,0]):, :]
#    img1, bordesup = borde(sup)
#    
#    inf = I[int(x[1,2,1]): int(x[1,3,1]), int(x[1,2,0]):, :]
#    img2, bordeinf = borde(inf)
#    
#    # Tomo un margen de +20 y -20 arriba y abajo para evitar posibles errores
#    inter = I[int(x[0,2,1]) + bordesup[0,0,1] + 20: int(x[1,2,1]) + bordeinf[0,0,1] - 20, int(x[0,2,0]): , :]
#    img3, bordeder = borde(inter)
#    
#    cuadro = I[int(x[0,2,1]) + np.max(bordesup[:,0,1]): int(x[1,2,1]) + np.min(bordeinf[0,0,1]), int(x[0,2,0]): int(x[0,2,0]) + np.min(bordeder[:,0,0]) , :]
#    
#    return cuadro


def det_frame(I, x):
    
    # Asumiendo que la primer perforación que llega es la superior
    sup = I[int(x[1,2,1]): int(x[1,3,1]), int(x[1,2,0]):, :]
    img1, bordesup = borde(sup)
    
    inf = I[int(x[0,2,1]): int(x[0,3,1]), int(x[0,2,0]):, :]
    img2, bordeinf = borde(inf)
    
    # Tomo un margen de +20 y -20 arriba y abajo para evitar posibles errores
    inter = I[int(x[1,2,1]) + bordesup[0,0,1] + 20: int(x[0,2,1]) + bordeinf[0,0,1] - 20, int(x[1,2,0]): , :]
    img3, bordeder = borde(inter)
    
    cuadro = I[int(x[1,2,1]) + np.max(bordesup[:,0,1]): int(x[0,2,1]) + np.min(bordeinf[0,0,1]), int(x[1,2,0]): int(x[1,2,0]) + np.min(bordeder[:,0,0]) , :]
    
    return cuadro
    

#plt.figure()
#plt.subplot(2,3,1)
#frame = cv2.imread(film_1[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(FillHoles1(gray), cmap = 'gray')
#
#plt.subplot(2,3,2)
#frame = cv2.imread(film_2[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(FillHoles1(gray), cmap = 'gray')
#
#plt.subplot(2,3,3)
#frame = cv2.imread(film_3[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(FillHoles1(gray), cmap = 'gray')
#
#plt.subplot(2,3,4)
#frame = cv2.imread(film_4[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(FillHoles1(gray), cmap = 'gray')
#
#plt.subplot(2,3,5)
#frame = cv2.imread(film_5[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(FillHoles1(gray), cmap = 'gray')

###
#plt.figure()
#plt.subplot(2,3,1)
#frame = cv2.imread(film_1[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(corners(fill_holes(gray)), cmap = 'gray')
#
#plt.subplot(2,3,2)
#frame = cv2.imread(film_2[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(corners(fill_holes(gray)), cmap = 'gray')
#
#plt.subplot(2,3,3)
#frame = cv2.imread(film_3[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(corners(fill_holes(gray)), cmap = 'gray')
#
#plt.subplot(2,3,4)
#frame = cv2.imread(film_4[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(corners(fill_holes(gray)), cmap = 'gray')
#
#plt.subplot(2,3,5)
#frame = cv2.imread(film_5[0], 1)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(corners(fill_holes(gray)), cmap = 'gray')
#plt.show()

###
#plt.figure()
#plt.subplot(2,3,1)
#frame = cv2.imread(film_1[0], 1)
#plt.imshow(orient(frame))
#
#plt.subplot(2,3,2)
#frame = cv2.imread(film_2[0], 1)
#plt.imshow(orient(frame))
#
#plt.subplot(2,3,3)
#frame = cv2.imread(film_3[0], 1)
#plt.imshow(orient(frame))
#
#plt.subplot(2,3,4)
#frame = cv2.imread(film_4[0], 1)
#plt.imshow(orient(frame))
#
#plt.subplot(2,3,5)
#frame = cv2.imread(film_5[0], 1)
#plt.imshow(orient(frame))
#plt.show()
    

## Con ginput encontré estas esquinas (las que están al lado de los frames)
#x = np.array([(416.05131964809397, 68.33896468188198),
# (416.05131964809397, 148.07521720915838),
# (412.7289757927908, 468.6813992459154),
# (414.39014772044237, 551.7399956284949)])
#    
#frame = cv2.imread(film_3[0], 1)
#frame = orient(frame)
#
#plt.figure()
#plt.imshow(det_frame(frame, x))
#
#
#y = np.array([(404.9354838709677, 297.1870967741936),
# (402.17741935483866, 382.6870967741936),
# (402.17741935483866, 697.1064516129031),
# (399.41935483870964, 788.1225806451612)])
#    
#frame = cv2.imread(film_4[0], 1)
#frame = orient(frame)
#
#plt.figure()
#plt.imshow(det_frame(frame, y))


'''cv2.imwrite('./imagenes/nuevo/' + file[0], frame)'''

#corners(orient(frame, find_perf(gray)))
#perf, esquinas = find_perf(gray)
#frame, esquinas = orient(frame, perf, esquinas)
#frame = corners(frame, esquinas)

#frame = cv2.imread(film_1[0], 1)
#
#perf, esquinas = find_perf(frame)
#frame = orient(perf)
#perf, esquinas = find_perf(frame)
#ang_mean = calc_rotation(esquinas)
#frame = rotateImage(frame, ang_mean)
#perf, esquinas = find_perf(frame)
#frame = det_frame(frame, esquinas)

###
plt.figure()
plt.subplot(2,3,1)
frame = cv2.imread(film_2[0], 1)
perf, esquinas = find_perf(frame)
frame = orient(frame, perf)
perf, esquinas = find_perf(frame)
ang_mean = calc_rotation(esquinas)
frame = rotateImage(frame, ang_mean)
perf, esquinas = find_perf(frame)
frame = det_frame(frame, esquinas)
plt.imshow(frame, cmap = 'gray')

plt.subplot(2,3,2)
frame = cv2.imread(film_2[0], 1)
perf, esquinas = find_perf(frame)
frame = orient(frame, perf)
perf, esquinas = find_perf(frame)
ang_mean = calc_rotation(esquinas)
frame = rotateImage(frame, ang_mean)
perf, esquinas = find_perf(frame)
frame = det_frame(frame, esquinas)
plt.imshow(frame, cmap = 'gray')

plt.subplot(2,3,3)
frame = cv2.imread(film_3[0], 1)
perf, esquinas = find_perf(frame)
frame = orient(frame, perf)
perf, esquinas = find_perf(frame)
ang_mean = calc_rotation(esquinas)
frame = rotateImage(frame, ang_mean)
perf, esquinas = find_perf(frame)
frame = det_frame(frame, esquinas)
plt.imshow(frame, cmap = 'gray')

plt.subplot(2,3,4)
frame = cv2.imread(film_4[0], 1)
perf, esquinas = find_perf(frame)
frame = orient(frame, perf)
perf, esquinas = find_perf(frame)
ang_mean = calc_rotation(esquinas)
frame = rotateImage(frame, ang_mean)
perf, esquinas = find_perf(frame)
frame = det_frame(frame, esquinas)
plt.imshow(frame, cmap = 'gray')

plt.subplot(2,3,5)
frame = cv2.imread(film_5[0], 1)
perf, esquinas = find_perf(frame)
frame = orient(frame, perf)
perf, esquinas = find_perf(frame)
ang_mean = calc_rotation(esquinas)
frame = rotateImage(frame, ang_mean)
perf, esquinas = find_perf(frame)
frame = det_frame(frame, esquinas)
plt.imshow(frame, cmap = 'gray')
plt.show()