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
from os.path import isfile, join
import os

#######################################################################################


def video_write(film_number, fps):
    if os.path.exists('./imagenes/' + film_number + '/film/film.avi'):
            os.remove('./imagenes/' + film_number + '/film/film.avi')

    if not os.path.exists('./imagenes/' + film_number + '/film/'):
       os.makedirs('./imagenes/' + film_number + '/film/')

    path_in= './imagenes/' + film_number + '/movie/'
    path_out = './imagenes/' + film_number + '/film/film.avi'
    video_create(path_in, path_out, fps)

    if os.path.exists('./imagenes/' + film_number + '/film/aligned_film.avi'):
            os.remove('./imagenes/' + film_number + '/film/aligned_film.avi')

    path_in2= './imagenes/' + film_number + '/aligned/'
    path_out2 = './imagenes/' + film_number + '/film/aligned_film.avi'
    video_create(path_in2, path_out2, fps)


    return
#######################################################################################

def video_create(path_in, path_out, fps):
    frame_array = []
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f))] #for sorting the file names properly
    #files.sort(key = lambda x: x[5:-4])
    #files.sort()

    for i in range(len(files)):
        filename=path_in + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)

        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(path_out,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    return
#######################################################################################
