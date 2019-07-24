#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:17:20 2019

@author: daiana
"""

import cv2
import os


#git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg

carpeta = 'film_2'

files = [img for img in os.listdir('./imagenes/' + carpeta + '/movie/') if img.endswith('.jpg')]
files.sort()

frame = cv2.imread('./imagenes/' + carpeta + '/movie/' + files[0], 1)
h, w, l = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
video = cv2.VideoWriter('movie.avi', fourcc, 20.0, (w, h))

for file in files:
    
    video.write(cv2.imread('./imagenes/' + carpeta + '/movie/' + file, 1))

video.release()
