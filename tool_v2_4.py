#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:20:01 2018

1. add thinning

@author: liuzewen
"""

import cv2
import numpy as np
from skimage.measure import label
from skimage.graph import route_through_array
from scipy.signal import convolve2d

ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
kernel = np.ones((3,3),np.uint8)

# sample the input image, each point would be represent by a patch whose center
# is itself.
def batching(img, batchSize, mode = None):
    # input a batch and output a vector set
    row, column = img.shape
    margin = batchSize/2
    newMap = np.zeros((row-batchSize+1, column-batchSize+1, batchSize*batchSize))
    for i in xrange(batchSize):
        if (-batchSize+i%batchSize+1) != 0:
            for j in xrange(batchSize):
                if (-batchSize+j%batchSize+1) != 0:
                    newMap[:, :, batchSize*i+j] = img[i:-batchSize+i%batchSize+1, j:-batchSize+j%batchSize+1]
                else:
                    newMap[:, :, batchSize*i+j] = img[i:-batchSize+i%batchSize+1, j:]
        else:
            for j in xrange(batchSize):
                if (-batchSize+j%batchSize+1) != 0:
                    newMap[:, :, batchSize*i+j] = img[i:, j:-batchSize+j%batchSize+1]
                else:
                    newMap[:, :, batchSize*i+j] = img[i:, j:]
    if mode == 1:
        newMap = newMap.reshape(((row-2*margin)*(column-2*margin), batchSize*batchSize))    
    return newMap



def smooth_length(length, interval):
    # smooth the output curve by the mean value of N closest neighbors
    smooth_length = []
    for i in xrange(interval, len(length)-len(length)%interval ,interval):
        smooth_length = smooth_length + [np.mean(length[i-interval:i+interval])]
    return smooth_length



def find_tip(img,startP): 
    # find connected ridge domian
    #img = cv2.GaussianBlur(img,(5,5),0)
    img = ridge_filter.getRidgeFilteredImage(img)
    #img = cv2.GaussianBlur(img,(5,5),0)
    #img = ridge_filter.getRidgeFilteredImage(img)
    #img = cv2.dilate(img,kernel,iterations = 1)
    # remove the unrelated domian
    ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #        cv2.THRESH_BINARY,7,2)
    
    # fix the breaks
    img = cv2.dilate(img,kernel,iterations = 1)
    
    # locate and separate the synapse which contains the given start tips
    img = label(img, connectivity = 2)
    synapse_label = img[startP]
    if synapse_label == 0:
        return startP, img
    img = img/synapse_label
    img[img != 1] = 100000                   
    
    # draw a boundary, in case the indicator escape
    img[0, :] = img[:, -1] = img[-1, :] = img[:, 0] = 100000
    img[startP] = 0
    
    # initial
    vistedmap = np.zeros(img.shape)
    
    distanceMap = vistedmap+100000
    distanceMap[img == 0] = 0
    temp = np.copy(distanceMap)
    
        
    # find the shortest path
    while (np.min(temp) != 100000):
        row, col = np.where(temp == np.min(temp))
        row = row[-1]
        col = col[-1]
        vistedmap[(row, col)] = 1
        temp[(row, col)] = 100000
        if distanceMap[(row, col+1)] > img[(row, col+1)]+distanceMap[(row, col)]:
            distanceMap[(row, col+1)] = img[(row, col+1)]+distanceMap[(row, col)]
            temp[(row, col+1)] = img[(row, col+1)]+distanceMap[(row, col)]
        if distanceMap[(row, col-1)] > img[(row, col-1)]+distanceMap[(row, col)]:
            distanceMap[(row, col-1)] = img[(row, col-1)]+distanceMap[(row, col)]
            temp[(row, col-1)] = img[(row, col-1)]+distanceMap[(row, col)]
        if distanceMap[(row+1, col)] > img[(row+1, col)]+distanceMap[(row, col)]:
            distanceMap[(row+1, col)] = img[(row+1, col)]+distanceMap[(row, col)]
            temp[(row+1, col)] = img[(row+1, col)]+distanceMap[(row, col)]
        if distanceMap[(row-1, col)] > img[(row-1, col)]+distanceMap[(row, col)]:
            distanceMap[(row-1, col)] = img[(row-1, col)]+distanceMap[(row, col)]
            temp[(row-1, col)] = img[(row-1, col)]+distanceMap[(row, col)]
    distanceMap[distanceMap == 100000] = 0
    tips = np.where(distanceMap == np.max(distanceMap))
    tips = np.array(tips).T.tolist()
    
    # find the best end tip candidate
    if len(tips) > 1:
        cost= np.array([route_through_array(img, startP, tip)[1] for tip in tips]).T
        realTip = tips[np.argmax(cost)]
        '''
        minCost = route_through_array(img, startP, tips[0])[1]
        for tip in tips[1:]:
            cost = route_through_array(img, startP, tip)[1]
            if cost < minCost:
                minCost = cost
                realTip = tip
        '''
    else:
        realTip = tips[0]
    return realTip, img


def getKernel(num_size = 5, num_angle = 6, num_lambda = 3, num_gamma = 1):
    # get every possible combination of parameters and generate their counterpart kernel
    set_size = [11+8*i for i in xrange(num_size)]
    set_theta = [i*(np.pi/num_angle) for i in xrange(num_angle)]
    set_lambd = [2*i+10-(num_lambda/2) for i in xrange(num_lambda)]
    set_gamma = 0.5
    g_kernel = []
    for size in set_size:
        for theta in set_theta:
            for lambd in set_lambd:
                g_kernel.append(cv2.getGaborKernel((size, size), 6.0, theta, lambd, set_gamma, 0, ktype = cv2.CV_32F))
    return g_kernel, set_size

# extract the gabor filter results of input image
def extractFeatures(img, num_size = 5, num_angle = 6, num_lambda = 3, num_gamma = 1):
    m, n = img.shape
    
    # get every possible combination of parameters and generate their counterpart kernel
    set_size = [11+8*i for i in xrange(num_size)]
    set_theta = [i*(np.pi/num_angle) for i in xrange(num_angle)]
    set_lambd = [2*i+10-(num_lambda/2) for i in xrange(num_lambda)]
    set_gamma = 0.5
    g_kernel = []
    for size in set_size:
        for theta in set_theta:
            for lambd in set_lambd:
                g_kernel.append(cv2.getGaborKernel((size, size), 6.0, theta, lambd, set_gamma, 0, ktype = cv2.CV_32F))
    
    # use Gabor keneral to generate feature map 
    temp = np.array([cv2.filter2D(img, cv2.CV_32F, g_kernel[i]) for i in xrange(len(g_kernel))])
    
    # adjust the data stucture
    feature_map = np.zeros((m, n, len(g_kernel)))
    for i in xrange(m-1):
        for j in xrange(n):
            feature_map[i, j, :] = temp[:, i, j]
            
    # Regard grey value as an attribute as well 
    img = np.reshape(img, (m, n, 1))
    feature_map = np.concatenate((feature_map, img), axis = 2)
    return feature_map, g_kernel

# Proportion Filter for optimizing the classification of binary nodes according 
# to their neighbor
def proportionFilter(img, ksize, ratio, binary = None):
    #ratio = 1-ratio
    
    # In order to fix the size change, give img a 0 border, the width of this 
    # border equals to the half of kernel size.
    border = ksize/2
    m, n = img.shape
    temp = np.zeros((m, border))
    img = np.concatenate((temp, img, temp), axis = 1)
    temp = np.zeros((border, n+2*border))
    img = np.concatenate((temp, img, temp), axis = 0)
    
    # logic filter
    temp = batching(img, ksize)
    img = np.sum(temp, axis = 2)
    #img[img<ratio*(ksize**2)] = 0
    # if binary switch on, return binary image
    if binary == 'binary':
        img[img != 0] = 1
    return img
    
# give dataset a label on the last unit of each record
def addLabel(src, label_number):
    src = np.array(src)
    temp = np.reshape(np.repeat(label_number, len(src)), (len(src), 1))
    src = np.concatenate((src, temp), axis = 1)
    return src  

# sample patches according to given coordinates 
def takePatch(img, coordinates, scale):
    m, n = img.shape
    radius = scale/2
    count = 0
    for i in xrange(len(coordinates)):
        i = i-count
        row, cal = coordinates[i]
        if row-radius<0 or row+radius+1 >= m-1 or cal-radius<0 or cal+radius+1 >= n-1:
            count = count+1
            coordinates = np.delete(coordinates, i, 0)
    patches = [img[row-radius:row+radius+1, cal-radius:cal+radius+1] for row, cal in coordinates]
    return coordinates, patches

# estimate the new coordinates when the image are rotated
def rotateCoordinates(m, n, angel, coordinates):
    coordinates = ([m, 0]-coordinates)*[1,-1]-[m/2, n/2]
    temp = []
    pi = np.pi
    angel = angel*pi/180
    for y, x in coordinates:
        l = np.sqrt(np.power(y, 2)+np.power(x,2))
        if x*y>0:
            if x>0:
                y = (m/2)-(l*np.sin(angel+np.arcsin(y/l)))
                x = (n/2)+(l*np.cos(angel+np.arccos(x/l)))
            else:
                y = (m/2)-(l*np.sin(angel+pi-np.arcsin(y/l)))
                x = (n/2)+(l*np.cos(angel-np.arccos(x/l)))
        elif x*y<0:
            if x>0:
                y = (m/2)-(l*np.sin(angel+np.arcsin(y/l)))
                x = (n/2)+(l*np.cos(angel-np.arccos(x/l)))
            else:
                y = (m/2)-(l*np.sin(angel+pi-np.arcsin(y/l)))
                x = (n/2)+(l*np.cos(angel+np.arccos(x/l)))
        elif x*y == 0:
            y = (m/2)-(l*np.sin(angel+np.arcsin((y+1)/l)))
            x = (n/2)+(l*np.cos(angel+np.arccos((x+1)/l)))
        if y>0 and y<m-0.5 and x>0 and x<n-0.5:
            temp.append([y, x])
    temp = np.rint(temp)
    
    return temp.astype('int')
        
# give original image and its target coordinate, and return the patch after 
# being totated 
def takeRotatedPatch(img, coordinates, scale, angel):
    m, n = img.shape
    coordinates = rotateCoordinates(m, n, angel, coordinates)
    M = cv2.getRotationMatrix2D((n/2, m/2), angel, 1)
    img = cv2.warpAffine(img,M,(n,m))
    return takePatch(img, coordinates, scale)

# extract features but with constant sampling size
    '''
def extractFeatures2(size, patch_angel, g_kernel, coordinates, img):
    (coordinates, patches) = takeRotatedPatch(img, coordinates, size, patch_angel)
    #print (np.array(patches)).shape
    # get every possible combination of parameters and generate their counterpart kernel
    feature_map = []
    # use Gabor keneral to generate feature map 
    for patch in patches:
        #print g_kernel
        feature_map.append(np.array([convolve2d(patch, kernel, mode='valid') for kernel in g_kernel]))
    
            
    # Regard grey value as an attribute as well
    temp = []
    print len(coordinates)
    for i in coordinates:
        temp.append(img[tuple(i)])
    print np.array(feature_map).shape
    print np.array(temp).shape
    feature_map = np.concatenate((feature_map, np.array(temp).reshape((len(coordinates), 1))), axis = 1)
    return feature_map
'''
def extractFeatures2(scale, angel, g_kernel, coordinates, img):
    m, n = img.shape
    coordinates = rotateCoordinates(m, n, angel, coordinates)
    M = cv2.getRotationMatrix2D((n/2, m/2), angel, 1)
    img = cv2.warpAffine(img,M,(n,m))
    radius = scale/2
    count = 0
    for i in xrange(len(coordinates)):
        i = i-count
        row, cal = coordinates[i]
        if row-radius<0 or row+radius+1 >= m-1 or cal-radius<0 or cal+radius+1 >= n-1:
            count = count+1
            coordinates = np.delete(coordinates, i, 0)
    patches = [img[row-radius:row+radius+1, cal-radius:cal+radius+1] for row, cal in coordinates]
    #print (np.array(patches)).shape
    # get every possible combination of parameters and generate their counterpart kernel
    feature_map = []
    # use Gabor keneral to generate feature map 
    for patch in patches:
        #print g_kernel.shape
        feature_map.append(np.array([convolve2d(patch, kernel, mode='valid') for kernel in g_kernel]))
    
    # Regard grey value as an attribute as well
    #temp = []
    #print len(coordinates)
    #for i in coordinates:
        #temp.append(img[tuple(i)])
    #print np.array(feature_map).shape
    #temp = np.array(temp).reshape((len(coordinates), 1))
    #print temp.shape
    #feature_map = np.concatenate((np.array(feature_map).reshape(((len(coordinates), len(g_kernel)))), temp), axis = 1)
    #return feature_map
    return np.array(feature_map).reshape(((len(coordinates), len(g_kernel))))

    
    
    
    
    
    
    
    
    
    