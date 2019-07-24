#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 23:15:10 2019

@author: daiana
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np


## Plotear los histogramas de lineas que pasan por las perforaciones
frame = cv2.imread('./imagenes/film_2/000819.jpg',cv2.IMREAD_GRAYSCALE)

# Remover ruido
frame = cv2.GaussianBlur(frame, (3,3), 0)

# Bordes Laplaciano
edges = cv2.Laplacian(frame, -1, ksize = 5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#edges = cv2.dilate(edges, kernel)

plt.figure()
plt.imshow(edges, cmap = 'gray')

_,threshold = cv2.threshold(edges,250,255,cv2.THRESH_BINARY)
plt.figure()
plt.imshow(threshold, cmap = 'gray')
_,contours,_ = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img2 = np.zeros(frame.shape)
#img2 = img.copy()*0


for cnt in contours:
	rect = cv2.minAreaRect(cnt)
	# box = vertices del rect sent antihorario arranca 
	# esquina superior derecha
	box = np.int0(cv2.boxPoints(rect)) 
	#filtro por area y long de los lados
	h,w = box[3,0]-box[0,0], box[3,1]-box[2,1]
	#print(h) 
	if (int(h*w) > 500) and (h>30):
		cv2.drawContours(img2,[box],0,(255),3)
	'''
	approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True),True)
	cv2.drawContours(img2, [approx],0,(255),1)
	#x = approx.ravel()[0]
	#y = approx.ravel()[1]
	#print(x,y)
	'''

plt.figure()
plt.imshow(img2,cmap='gray')
plt.show()



'''
plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(threshold)
plt.show()

h,w = img.shape
'''

'''
## valores en la linea de las perforaciones:
h_vect = np.arange(h)
line = 1250

plt.figure()
plt.plot(h_vect,img[:,line])
plt.grid(True)
plt.show() 
'''

