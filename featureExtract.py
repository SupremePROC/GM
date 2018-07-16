#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:30:22 2018

give coordinates, then extract the features of these points

@author: liuzewen
"""

import cv2
import numpy as np
from os import listdir
import tool_v2_3 as tool

path = 'data/s3m0v283/'
img = cv2.imread(path+'test.tif', 0)
feature = []

axon = np.load(path+'coordinates_axon.npy')

num_size = 8
# orientations
num_angle = 6
# wavelengths
num_lambda = 3
# wavelength ratios
num_gamma = 1
feature_map, g_kernel = tool.extractFeatures(img, num_size, num_angle, num_lambda, num_gamma)
for i in axon:
    feature.append(feature_map[i[0], i[1], :])
    
np.save(path+'feature_axon', np.array(feature))

