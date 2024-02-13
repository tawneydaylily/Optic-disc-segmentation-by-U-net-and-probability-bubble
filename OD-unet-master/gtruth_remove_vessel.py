# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 21:10:34 2017

@author: pc
"""

import os
import cv2

threshold=200

vessel_dir=os.listdir("./DRIVE/training/vessel/")
manual_dir=os.listdir("./DRIVE/training/1st_manual/")

for src in manual_dir:
    print (src)
    manual = cv2.imread("./DRIVE/training/1st_manual/"+src)
    vessel = cv2.imread("./DRIVE/training/vessel/"+src[20:src.find('.')]+".tif.png")
    for i in range(len(manual)):
        for j in range(len(manual[0])):
            if vessel[i,j,0]>threshold:
                manual[i,j,0]=0
                manual[i,j,1]=0
                manual[i,j,2]=0
                
#    cv2.imshow("",manual)
#    cv2.waitKey(0)
    
    cv2.imwrite("./DRIVE/training/remove_vessel/"+src,manual)

