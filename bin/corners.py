#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################
import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread('./imagenes/film_2/000819.jpg',cv2.IMREAD_GRAYSCALE)

I = np.float32(I)

'''
for i in range(I.shape[0]):
	for j in range(I.shape[1]):
		if I[i,j] < 200:
			I[i,j] = 0
'''
_,I = cv2.threshold(I,200,255,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
I = cv2.erode(I,kernel,iterations = 2)
kernel = np.ones((4,4),np.uint8)
I = cv2.dilate(I,kernel, iterations = 2)

#plt.figure()
#plt.imshow(I, cmap = 'gray')

'''
######################################

_,threshold = cv2.threshold(I,200,255,cv2.THRESH_BINARY)
plt.figure()
plt.imshow(threshold, cmap = 'gray')

_,contours,_ = cv2.findContours(threshold, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img2 = np.zeros(I.shape)
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
######################################

'''

corners = cv2.goodFeaturesToTrack(I,40,0.01,50)
corners = np.int0(corners)


for corner in corners:
	x, y = corner.ravel()
	cv2.circle(I, (x,y), 3, 150, -1)

plt.figure()
plt.imshow(I,cmap='gray')
plt.show()