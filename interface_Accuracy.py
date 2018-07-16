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


dir_name = '/Users/liuzewen/Documents/Unversity/Master project/GM/step1/movie3/a/'
filenames = listdir(dir_name)
filenames.sort()
filenames.reverse()
filenames = filenames[:-2]
filenames = filenames[:168]
numFrames = len(filenames)
i = 0
#指定图像文件名称
background_image_filename = dir_name+filenames[i]
img = cv2.imread(background_image_filename, 0)
row, col = img.shape
 

 
pygame.init() #初始化pygame,为使用硬件做准备
screen = pygame.display.set_mode((col, row), 0, 32)
#创建了一个窗口
pygame.display.set_caption("Please click start and end points")
#设置窗口标题

background = pygame.image.load(background_image_filename).convert()

newTips = []

while True:

    background_image_filename = dir_name+filenames[i]
    img = cv2.imread(background_image_filename, 0)
    row, col = img.shape
    background = pygame.image.load(background_image_filename).convert()
    screen.blit(background, (0,0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # quit when finish
            np.savetxt('accuracy/test_a/human.txt', newTips)
            pygame.quit()
            exit()
            
        if event.type == MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            newTips = newTips+[[y, x]]
            print filenames[i]
            i+=1
            print (y, x)

    if i == numFrames:
        np.savetxt('accuracy/test_a/human.txt', newTips)
        pygame.quit()
        exit()
    pygame.display.update()






