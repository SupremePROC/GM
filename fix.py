#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 02:40:01 2018

this script is for fixing the breaks along the axon main branches

@author: liuzewen
"""

import cv2
import numpy as np
from os import listdir

filenames = listdir('image/movie0_sk/')
filenames.sort()
filenames.reverse()
filenames = filenames[:-1]

path1 = 'image/movie0_sk/'
path2 = 'image/movie0_sk2/'

for filename in filenames:
    img1