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




count = 0

img = cv2.imread('image/test.tif', 0)
row, col = img.shape

save_path = 'data/s3m0v0/'

 
pygame.init() #初始化pygame,为使用硬件做准备
screen = pygame.display.set_mode((col, row), 0, 32)
#创建了一个窗口
pygame.display.set_caption("Please click start and end points")
#设置窗口标题
background = pygame.image.load('image/test.tif').convert() 

categroy = save_path+'coordinates_axon.npy'
#categroy = save_path+'coordinates_cellbody.npy'
#categroy = save_path+'coordinates_blob.npy'
#categroy = save_path+'coordinates_background.npy'
#categroy = save_path+'coordinates_dirts.npy'

data = []
#data = list(np.load(categroy)) 

while True:
#游戏主循环


    screen.blit(background, (0,0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            #接收到退出事件后退出程序
            np.save(categroy, np.array(data))
            pygame.quit()
            exit()
            
        if event.type == MOUSEBUTTONDOWN:
            cal, row = pygame.mouse.get_pos()
            #background[row-3:row, cal-3:row] = 0
            data.append([cal, row])
            print (row, cal)
            count= count+1
            print count

    pygame.display.update()






