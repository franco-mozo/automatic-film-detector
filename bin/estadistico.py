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


def perf_matching(I, found_perf, perf_corners, tmp_corners):

    corn_matrix, ang = standarize_order(found_perf, perf_corners)

    # cantidad de perforaciones en la matriz cubo (menor de las dos)
    index = 0
    i = 0
    while perf_corners[i,0,0] != -1:
        i += 1
    j = 0
    while tmp_corners[j,0,0] != -1:
        j += 1
    if i <= j:
        index = i
    else:
        index = j


    # armo las matrices src_points, dst_points
    src_points = np.zeros( (index*4 ,2 ))#, dtype=np.uint8) #(corn_matrix)
    dst_points = np.zeros( (index*4 ,2 ))#, dtype=np.uint8) #(tmp_corners)

    j = 0
    for i in range(0, index*4, 4):
        src_points[i:i+4, :] = corn_matrix[j, :, :].copy()
        dst_points[i:i+4, :] = tmp_corners[j, :, :]
        j += 1

    M, _ = cv2.findHomography(src_points,dst_points)
    if ang == 270:
        width, height = I.shape[0], I.shape[1]
    else:
        width, height = I.shape[1], I.shape[0]
    matched_im = cv2.warpPerspective(I, M, (width, height))
    #matched_im = cv2.warpAffine(I, M[], (I.shape[0], I.shape[1]) )

    return matched_im

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

################################# punto 5 #############################################


#######################################################################################

## GUARDO USANDO LA IDEA DE MATCHEAR CON HOMOGRAFIA
def main2():
    carpeta = 'film_2'

    files = os.listdir('./imagenes/' + carpeta + '/')

    first_im = cv2.imread('./imagenes/' + carpeta + '/' + files[0], 1)
    test_im = cv2.imread('./imagenes/' + carpeta + '/' + files[1], 1)

    # Tratamiento del template
    found_tmp, tmp_corners = find_perf(first_im)
    oriented_tmp, ang = orient(first_im, found_tmp)
    tmp_oriented_corners = orient_corners(oriented_tmp, tmp_corners, ang)

    if not os.path.exists('./imagenes/' + carpeta + '/movie/'):
       os.makedirs('./imagenes/' + carpeta + '/movie/')

    for file in files:
        # alineacion de la pelicula
        frame = cv2.imread('./imagenes/' + carpeta + '/' + file, 1)

        found_perf, perf_corners = find_perf(frame)
        #test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
        frame = perf_matching(frame, found_perf, perf_corners, tmp_oriented_corners )

        '''
        perf, esquinas = find_perf(frame)
        frame, ang = orient(frame, perf)
        esquinas = orient_corners(frame, esquinas, ang)

        try:
            frame = det_frame(frame, esquinas, ang)
        except:
            pass
        '''
        cv2.imwrite('./imagenes/' + carpeta + '/movie/' + file, frame)

    return



def align_images(src, template):

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    # Convierto a escala de gris.
    src_gs = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    template_gs = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Detecto features ORB and computo descriptores
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(src_gs, None)
    keypoints2, descriptors2 = orb.detectAndCompute(template_gs, None)

    # Matchear features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Ordeno por orden.
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Saco los peores
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Dibujo
    #im_matches = cv2.drawMatches(src, keypoints1, template, keypoints2, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)

    # Ubicacion de los best matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Calculo homografia
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    height, width, channels = template.shape
    src_reg = cv2.warpPerspective(src, h, (width, height))

    return src_reg, h

## GUARDO LA ORB
def main3():
    carpeta = 'film_3'

    files = os.listdir('./imagenes/' + carpeta + '/')
    mid = int(len(files)/2)
    first_im = cv2.imread('./imagenes/' + carpeta + '/' + files[mid], 1)

    # Tratamiento del template
    found_tmp, tmp_corners = find_perf(first_im)
    oriented_tmp, ang = orient(first_im, found_tmp)
    tmp_oriented_corners = orient_corners(oriented_tmp, tmp_corners, ang)

    if not os.path.exists('./imagenes/' + carpeta + '/movie/'):
       os.makedirs('./imagenes/' + carpeta + '/movie/')

    for file in files:
        # alineacion de la pelicula
        frame = cv2.imread('./imagenes/' + carpeta + '/' + file, 1)
        frame, h = align_images(frame, oriented_tmp)

        cv2.imwrite('./imagenes/' + carpeta + '/movie/' + file, frame)
    return

## GUARDO LAS PERFORACIONES ENCONTRADAS EN /MOVIE
def main4():
    carpeta = 'film_3'

    files = os.listdir('./imagenes/' + carpeta + '/')

    if not os.path.exists('./imagenes/' + carpeta + '/movie/'):
       os.makedirs('./imagenes/' + carpeta + '/movie/')

    for file in files:
        # alineacion de la pelicula
        frame = cv2.imread('./imagenes/' + carpeta + '/' + file, 1)

        found_perf, perf_corners = find_perf(frame)
        cv2.imwrite('./imagenes/' + carpeta + '/movie/' + file, found_perf)

    return


#main()
#main2()
#main3()
#main4()
