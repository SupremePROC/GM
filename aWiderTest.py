#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 23:37:41 2018

@author: liuzewen
"""

import cv2
import numpy as np
import time
from os import listdir
from tool_v2_4 import extractFeatures, proportionFilter, addLabel
from sklearn.ensemble import RandomForestClassifier
from skimage.morphology import skeletonize


filenames = listdir('image/movie0_axon_1/')
filenames.sort()
filenames.reverse()
filenames = filenames[:-1]

nodes = [[(387, 104), (185, 122)], [(351, 243), (513, 350)], [(55, 195), 
          (167, 49)], [(55, 194), (34, 130)], [(55, 228), (116, 313)], 
    [(3, 441), (293, 509)], [(358, 205), (392, 137)], [(509, 97), (450, 158)], 
    [(95, 686), (176, 691)], [(3, 344), (62, 331)], [(36, 654), (4, 673)], 
    [(40, 407), (63, 443)], [(58, 486), (114, 497)], [(212, 692), (188, 684)]]

temp = np.zeros((520, 694))
for i in nodes:
    for j in xrange(2):
        row, cal = i[j]
        temp[row-10:row+10, cal-10:cal+10] = 1
nodes = np.argwhere(temp == 1)
kernel = np.ones((3,3),np.uint8)
mask = np.zeros((m-2, n-2))
for filename in filenames:
    print filename
    # test
    a0 = cv2.imread('image/movie0_axon_result/'+filename, 0).astype('float32')/255
    m, n = a0.shape 
    

    result = np.copy(mask)
    for i in nodes:
        row, cal = i
        if row>=m-3 or row<0 or cal>n-3 or cal<0:
            continue
        b1 = cv2.floodFill(mask.astype('uint8'), a0.astype('uint8'), (cal-1, row-1), 255, cv2.FLOODFILL_MASK_ONLY)[1]
        result[b1 == 255] = 255
    
    cv2.imwrite(c+filename, result)
    
    skeleton = skeletonize(result/255)*255
    cv2.imwrite('image/movie0_sk2/'+filename, skeleton)
    
    '''
    img2 = cv2.imread('image/old/movie0_mark_2/'+filename, 0)
    img2 = img2[1:-1, 1:-1]
    blank = np.zeros(img2.shape)
    blank[img2 == 50] = 100
    blank[skeleton == 255] = 255
    cv2.imwrite('image/movie_ds/'+filename, blank)
    '''
    
#    connectedComponentsWithStats() 