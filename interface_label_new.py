#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 00:25:35 2018

@author: liuzewen
"""
import pygame #导入pygame库
from pygame.locals import *
import cv2
from sys import exit #向sys模块借一个exit函数用来退出程序
import numpy as np
from os import listdir
import tool_v2_3 as tool

path = 'data/s3m0v283/'
filename = path+'test.tif'

#path_feature = path+'feature_axon.npy'
#path_feature = path+'feature_cellbody.npy'
#path_feature = path+'feature_blob.npy'
path_feature = path+'feature_background.npy'
#path_feature = path+'feature_dirt.npy'

#path_coordinate = path+'coordinates_axon.npy'
#path_coordinate = path+'coordinates_cellbody.npy'
#path_coordinate = path+'coordinates_blob.npy'
path_coordinate = path+'coordinates_background.npy'
#path_coordinate = path+'coordinates_dirt.npy'



img = cv2.imread(filename, 0)
row, col = img.shape

 
pygame.init() #初始化pygame,为使用硬件做准备
screen = pygame.display.set_mode((col, row), 0, 32)
#创建了一个窗口
pygame.display.set_caption("Please click start and end points")
#设置窗口标题
background = pygame.image.load(filename).convert() 


#coordinates = []
#feature = []
coordinates = list(np.load(path_coordinate))
feature = list(np.load(path_feature)) 

num_size = 8
# orientations
num_angle = 6
# wavelengths
num_lambda = 3
# wavelength ratios
num_gamma = 1
feature_map, g_kernel = tool.extractFeatures(img, num_size, num_angle, num_lambda, num_gamma)

count = len(coordinates)
while True:
#游戏主循环
    screen.blit(background, (0,0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            #接收到退出事件后退出程序
            np.save(path_coordinate, np.array(coordinates))
            np.save(path_feature, np.array(feature))
            pygame.quit()
            exit()
            
        if event.type == MOUSEBUTTONDOWN:
            cal, row = pygame.mouse.get_pos()
            #background[row-3:row, cal-3:row] = 0
            coordinates.append([cal, row])
            print (row, cal)
            feature.append(feature_map[row, cal, :])

            count= count+1
            print count

    pygame.display.update()






