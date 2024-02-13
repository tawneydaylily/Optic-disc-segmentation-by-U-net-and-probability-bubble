# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:17:29 2017
方差
@author: pc
"""


print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.cluster import KMeans
from sklearn import datasets
import os
import csv
from numpy import mat
import random
import string

np.random.seed(5)

"""
高斯滤波核
"""
kernel=np.uint8(np.zeros((5,5)))
for x in range(5):
    kernel[x,2]=1;
    kernel[2,x]=1;


#"""
#反色
#"""
#def inverse_color(image):
#
#    height,width,temp = image.shape
#    img2 = image.copy()
#
#    for i in range(height):
#        for j in range(width):
#            img2[i,j] = (255-image[i,j][0],255-image[i,j][1],255-image[i,j][2]) 
#    return img2
"""
颜色近似度量
"""
def color_distance(c1,c2):
    return ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)**0.5

"""
距离公式
"""
def distance(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5
"""
去除血管
"""
def remove_vessel(image,vessel,threshold=200):

    height,width,temp = image.shape
    img2 = image.copy()
    for i in range(height):
        for j in range(width):
            if vessel[i,j][0]>threshold:
                img2[i,j]=(0,0,0)
    img_without_dilate=img2.copy()
    img2 = cv2.dilate(img2, kernel)
    img2 = cv2.dilate(img2, kernel)
    img2 = cv2.dilate(img2, kernel)
    img2 = cv2.erode(img2, kernel)
    img2 = cv2.erode(img2, kernel)
    img2 = cv2.erode(img2, kernel)
    
    
    return img2,img_without_dilate

count=0
remove_illu_Dir= os.listdir(os.path.dirname(os.getcwd())+'/VesselSegmentation/input/')
for src in remove_illu_Dir:
    count+=1
    print ("\npicture:",src,"(%s/%s)"%(str(count),str(len(remove_illu_Dir))))
    
    image = cv2.imread(os.path.dirname(os.getcwd())+'/VesselSegmentation/input/'+src)
    vessel = cv2.imread(os.path.dirname(os.getcwd())+'/VesselSegmentation/output/'+src+'.png')
    vessel=cv2.resize(vessel,(500,500))
    image=cv2.resize(image,(500,500))
    img,_=remove_vessel(image,vessel,threshold=200)
    
    
#    cv2.imshow(src,img)
#    cv2.waitKey(0)
    cv2.imwrite('./input/'+src,img)
