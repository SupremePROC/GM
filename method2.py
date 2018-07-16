#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 16:05:21 2018

This scripts is for generating the graph of this branch structure and 
abandoning useless edges.

the material of this step is the output of method1 where interest points
labelled. Before using it, we change the start tips to a branch points, making
the connect relationship easier(there is only branchpoint-tips edge existing
now).

@author: liuzewen
"""

import numpy as np
import cv2
import tool_v2_4 as tool

from sklearn.feature_extraction import image


img = cv2.imread('t6.png', 0)/255
m, n = img.shape
img_copy = image.extract_patches_2d(img, (3, 3))
sum_img = np.sum(img_copy.reshape((m-2)*(n-2), 9), axis = 1)
sum_img = sum_img.reshape((m-2, n-2)).astype('int')


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

'''
class node:
    neighbor = []
    edge = []
    def __init__(self, index, coordinates, axon, label):
        self.index = index
        self.coordinates = coordinates
        self.axon = axon
        self.label = label

class edge:
    length = 0
    weight = 0 
    def __init__(self, index, coordinates, node):
        self.index = index
        self.coordinates = coordinates
        self.node = node
        
my_node = []
my_edge = []
'''   
        
# use dictionary to store the graph information. Due to possible abandonmrnt
#later, give every node an index as their key value.


my_node = {'index':[], 'coordinates':[], 'neighbor':[], 'edge':[], 'axon':[], 'class':[]}
my_edge = {'index':[], 'tip':[], 'path':[], 'length':[], 'weight':[], 'axon':[]}

img3 = np.copy(img)
cnt_com_all = cv2.connectedComponentsWithStats(img)[1]-1
start_point = [8, 20]

nodes = np.argwhere(img >= 2)
temp = 0

# initialize nodes
for row, cal in np.argwhere(img >= 2):
    my_node['coordinates'].append([row, cal])
    my_node['index'].append(temp)
    temp = temp+1
    my_node['class'].append(int(img[row, cal]))
    my_node['neighbor'].append([])
    my_node['edge'].append(-1)
    my_node['axon'].append(cnt_com_all[row, cal])

img[img>1]=0

cnt_com = cv2.connectedComponentsWithStats(img)
edge_img = cnt_com[1]-1
node_number = len(my_node['coordinates'])
edge_number = cnt_com[0]-1

for i in xrange(node_number):
    row, cal = my_node['coordinates'][i]
    a = edge_img[row-1:row+2, cal-1:cal+2]
    my_node['edge'][i] = a[a>=0]

temp = 0
# initialize path
for i in xrange(edge_number):
    my_edge['index'].append(temp)
    temp = temp+1
    my_edge['tip'].append([])
    my_edge['path'].append(np.argwhere(edge_img == i))
    my_edge['length'].append(cnt_com[2][i+1][-1])

   
for i in xrange(node_number):
    temp = my_node['edge'][i]
    for j in temp:
        j = np.argwhere(np.array(my_edge['index']) == j)[0][0]
        my_edge['tip'][j].append(i)
    



img = cv2.imread('t3.png', 0)[2:-2, 2:-2]
img2 = img3
for i in my_edge['index']:
    temp = 0
    for row, cal in my_edge['path'][i]:
        temp = temp+img[row, cal]
    my_edge['weight'].append(temp/my_edge['length'][i])
    my_edge['axon'].append(cnt_com_all[row, cal])
    

axon_lim = range(cnt_com_all.max())
for i in axon_lim:
    axon_lim[i] = img[cnt_com_all == i]
            
            
for i in my_edge['index']:    
    if my_edge['weight'][i]>=60:
        for row, cal in my_edge['path'][i]:
            img2[row, cal] = 0
#img3 = cv2.imread('t2.png', 0)      
'''
for i in xrange(node_number):
    temp = np.array(my_edge['tip']) - i
    temp = temp[:, 0]*temp[:, 1]
    for j in xrange(2):
        temp = np.argwhere(my_edge['tip']==i)
'''
    
    
    
    
'''
for row, cal in nodes:
    my_node.append[node(node_index, [row, cal], 0, img[row, cal])]
    #if img[row, cal] == 1:
'''
    

    