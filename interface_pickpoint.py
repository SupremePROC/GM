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




img = cv2.imread('movie3/test.tif', 0)
row, col = img.shape

 
pygame.init() #初始化pygame,为使用硬件做准备
screen = pygame.display.set_mode((col, row), 0, 32)
#创建了一个窗口
pygame.display.set_caption("Please click start and end points")
#设置窗口标题
background = pygame.image.load('movie3/test.tif').convert() 

#categroy = 'data/v2/coordinates_positive.npy'
#
categroy = 'data/v2/coordinates_negative.npy'

#data = []
data = list(np.load(categroy)) 

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

    pygame.display.update()






