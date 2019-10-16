#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:08:49 2019

@author: shane
"""
import numpy as np
import matplotlib.pyplot as plt

def plotGraph(name, keyWord, color, label, lineStyle=None):
    acc = []
    with open(name, 'r') as f:
        for num, line in enumerate(f):
            if keyWord in line:
                acc.append(float(line.replace(keyWord, "")))#/1000000)
#    x_Axis = np.arange(len(acc))
    x_Axis = np.arange(100)
#    plt.plot(x_Axis, acc, color=color, label=label)
    if lineStyle is None:
        plt.plot(x_Axis, acc[0:100], color=color, label=label)
    else:
        plt.plot(x_Axis, acc[0:100], color=color, label=label, linestyle=lineStyle)
#keyWord = "FPS: "
#keyWord = "eval mean loss: "
#keyWord = "eval segmentation avg class acc: "
#keyWord = "eval box estimation accuracy (IoU=0.7): "
keyWord = "mean loss: "
#keyWord = "box estimation accuracy (IoU=0.7): "

def plotTrainingGraph(name, keyWord, color, label, lineStyle=None):
    acc = []
    with open(name, 'r') as f:
        for num, line in enumerate(f):
            if keyWord in line and "eval" not in line:
                acc.append(float(line.replace(keyWord, "")))
    acc = acc[::50]
    x_Axis = np.arange(330)
    acc = smooth(acc[:330], 9)
#    x_Axis = np.arange(100)
#    plt.plot(x_Axis, acc, color=color, label=label)
    if lineStyle is None:
        plt.plot(x_Axis, acc, color=color, label=label)
    else:
        plt.plot(x_Axis, acc, color=color, label=label, linestyle=lineStyle)
    
def smooth(a,windowSize):
  out0 = np.convolve(a,np.ones(windowSize,dtype=int),'valid')/windowSize
  r = np.arange(1,windowSize-1,2)
  start = np.cumsum(a[:windowSize-1])[::2]/r
  stop = (np.cumsum(a[:-windowSize:-1])[::2]/r)[::-1]
  return np.concatenate(( start , out0, stop ))

fig = plt.figure()
fig.suptitle(keyWord, fontsize=16)
plt.xlabel('Time')
plt.ylabel('Loss')
#plotGraph('/localhome/sxu/log_v2_DIM4/log_train.txt', keyWord, 'red', 'PointNet v2 with intensity')
#plotGraph('/localhome/sxu/log_tiny1_100/log_train.txt', keyWord, 'red', 'PointNet v1 Tiny')
#plotTrainingGraph('/localhome/sxu/log_tiny2_newLoss2/log_train.txt', keyWord, 'red', 'MyPointNet_V2 2')
plotTrainingGraph('/localhome/sxu/log_tiny2_100/log_train.txt', keyWord, 'red', 'OurFPN')
#plotTrainingGraph('/localhome/sxu/log_tiny2_newLoss05/log_train.txt', keyWord, 'green', 'MyPointNet_V2 0.5')
#plotGraph('/localhome/sxu/log_v1_tiny3/log_train.txt', keyWord, 'blue', 'MyPointNet_V1')
#plotGraph('/localhome/sxu/log_mini_100/log_train.txt', keyWord, 'red', 'mini1')
#plotGraph('/localhome/sxu/log_mini2_100/log_train.txt', keyWord, 'pink', 'mini2')
#plotGraph('/localhome/sxu/log_v2_MSG/log_train.txt', keyWord, 'red', 'PointNet_V2_MSG')
plotTrainingGraph('/localhome/sxu/log_v2_SSG/log_train.txt', keyWord, 'green', 'F-PN v2') # without intensity
#plotGraph('/localhome/sxu/log_v1_DIM4/log_train.txt', keyWord, 'blue', 'PointNet v1 with intensity', '--')
plotTrainingGraph('/localhome/sxu/log_org100/log_train.txt', keyWord, 'blue', 'F-PN v1')
#plotGraph('/localhome/sxu/log_v2_DIM4/log_train.txt', keyWord, 'black', 'PointNet v2 with intensity')

#plotGraph('/localhome/sxu/Desktop/MA/frustum-pointnets-master/speed_test_org100_pc.txt', keyWord, 'black', 'PC_org100')
#plotGraph('/localhome/sxu/Desktop/MA/frustum-pointnets-master/speed_test_tiny1.txt', keyWord, 'red', 'PC_tiny1')
#plotGraph('/localhome/sxu/speed_test_orgv1.txt', keyWord, 'black', 'Original PointNet v1')
#plotGraph('/localhome/sxu/speed_test_tiny2.txt', keyWord, 'blue', 'MyPointNet_V1')
#plotGraph('/localhome/sxu/speed_test_tiny2.txt', keyWord, 'red', 'MyPointNet v2')
#plotGraph('/localhome/sxu/speed_test_tiny3.txt', keyWord, 'blue', 'MyPointNet v1')
#plotGraph('/localhome/sxu/speed_test_tiny1.txt', keyWord, 'red', 'PointNet v1 Tiny')
#plotGraph('/localhome/sxu/speed_test_summit.txt', keyWord, 'green', 'Summit XI', '--')
#plotGraph('/localhome/sxu/speed_test_jetson.txt', keyWord, 'green', 'Jetson TX2')
#plotGraph('/localhome/sxu/speed_test_v2SSG.txt', keyWord, 'green', 'PointNet_V2_SSG')
#plotGraph('/localhome/sxu/speed_test_v2MSG.txt', keyWord, 'red', 'PointNet_V2_MSG')

#plotGraph('/localhome/sxu/log_tiny2_100/log_train.txt', keyWord, 'black', 'gamma = 1')
#plotGraph('/localhome/sxu/log_tiny2_newLoss2/log_train.txt', keyWord, 'green', 'gamma = 2')
#plotGraph('/localhome/sxu/log_tiny2_newLoss05/log_train.txt', keyWord, 'red', 'gamma = 0.5')
plt.legend(loc='upper right')
#plt.savefig('AccComparison.png', dpi=600)
plt.savefig('LossComparison.png', dpi=600)
#plt.savefig('SpeedComparison.png', dpi=600)



