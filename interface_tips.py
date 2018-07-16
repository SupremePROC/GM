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
from skimage.graph import route_through_array

def highlight(img, nodes):
    #axon = np.load('data/s3m0v0/coordinates_axon.npy')
    axon = []
    for startP, endP in nodes:
        cost= np.array(route_through_array(img, startP, endP)[0])
        for i in cost:
            img[tuple(i)] = 255
            axon.append(i)
    return img, axon



save_path = 'data/s3m0v283/'
filename = save_path+'test.tif'

img = cv2.imread(filename, 0)
row, col = img.shape

 
pygame.init() #初始化pygame,为使用硬件做准备
screen = pygame.display.set_mode((col, row), 0, 32)
#创建了一个窗口
pygame.display.set_caption("Please click start and end points")
#设置窗口标题
background = pygame.image.load(filename).convert() 

tip_pair = []
key = 0
count = 1
while True:
#游戏主循环


    screen.blit(background, (0,0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            #接收到退出事件后退出程序
            np.save(save_path+'tips', np.array(tip_pair))


            img2, axon = highlight(img, tip_pair)
            cv2.imwrite(save_path+'temp.png', img2)
            np.save(save_path+'coordinates_axon.npy', axon)
            pygame.quit()
            exit()
            
        if event.type == MOUSEBUTTONDOWN:
            if key == 0:
                print str(count)+'# Start point:'
                cal, row = pygame.mouse.get_pos()
                startP = (row, cal)
                print startP
                key = key+1
            elif key == 1:
                print str(count)+'# End point:'
                cal, row = pygame.mouse.get_pos()
                endP = (row, cal)
                print endP
                key = key-1
                count = count+1
                

                tip_pair.append([startP, endP])
                

    pygame.display.update()






