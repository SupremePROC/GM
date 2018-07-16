#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 12:50:53 2018

This script is for detecting interest points.

@author: liuzewen
"""

import numpy as np
import cv2
from sklearn.feature_extraction import image
'''
class node:
    
    def __init__(neighbor, edge):
        pass
'''

my_graph = {'index':[], 'coordinates':[], 'neighbor':[], 'edge':[]}


img = cv2.imread('s238.tif', 0)/255
m, n = img.shape
img_copy = image.extract_patches_2d(img, (3, 3))
sum_img = np.sum(img_copy.reshape(356040, 9), axis = 1)
sum_img = sum_img.reshape((516, 690)).astype('int')


# having 2 neighbors means this node is a tip
sum_img[sum_img == 2] = -2
# having more than 4 neighbors means this node is a branch point
sum_img[sum_img >= 4] = -4
sum_img[sum_img >= 0] = 0
sum_img = sum_img*-1
img = img[1:-1, 1:-1]
sum_img[img == 0] = 0
img[sum_img == 2] = 2
img[sum_img == 4] = 3

#for i in img_copy:

#img = img*100

#img[sum_img == 2] = 255
#cv2.imwrite('t1.png', img)
