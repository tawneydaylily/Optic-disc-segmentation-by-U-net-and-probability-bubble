# -*- coding: utf-8 -*-
"""

"""

import cv2  
import numpy as np    
# import heatmap
import os
from PIL import Image

#filelist=np.array([])
# 遍历指定目录，显示目录下的所有文件名
open_count=0

# vessel_Dir =  os.listdir('./input/origin/')

"""
打开图片
"""
def OpenPic(src,Dir_str):
    src_found=""
    Dir = os.listdir(Dir_str)
    for i in range(len(Dir)):
        if Dir[i].find(src)>=0:# and Dir[i].find(".png")>0:
#            print src[:len(src)-8],"*",focus_Dir[i]
            src_found=Dir[i]
            break
        else:
            if i is range(len(Dir)-1):
                print (src,"PICTURE NOT FOUND!")
    print (src_found)
    return cv2.imread(Dir_str+src_found)
"""
画泡泡
"""
def DrawCircle(edges, threshold, minLineLength, maxLineGap):
    def equation_y(x):
        if not x2 - x1 == 0:
            return (x * (y2 - y1) + x1 * y1 - x1 * y2 - x1 * y1 + x2 * y1) / (float)(x2 - x1)
        else:
            return 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 10, threshold, minLineLength, maxLineGap)
    parameter_lines = []
    for i in range(len(lines)):
        #    for x1,y1,x2,y2 in lines[0]:
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        #    k=(y2-y1)/(float)(x2-x1)

        # ax+by=e
        a = y1 - y2
        b = x2 - x1
        e = x1 * y1 - x1 * y2 + x2 * y1 - x1 * y1
        #    equation_lines=np.append(equation_lines,[a,b,e])
        parameter_lines.append([a, b, e])
        #    print int(equation_y(0))
        if not equation_y(0) == float("inf"):
            #            cv2.line(img,(0,int(equation_y(0))),(len(img),int(equation_y(len(img)))),(0,255,0),2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    _, bubble = cv2.threshold(ori, 255, 255, cv2.THRESH_BINARY)
    CrossPoints = []
    for i in range(len(parameter_lines) - 1):
        for j in range(1, len(parameter_lines)):

            #        print parameter_lines[i],parameter_lines[j]#任意直线的两两组合
            a = parameter_lines[i][0]
            b = parameter_lines[i][1]
            e = parameter_lines[i][2]
            c = parameter_lines[j][0]
            d = parameter_lines[j][1]
            f = parameter_lines[j][2]
            #        print a,b,c,d,e,f
            CrossPoint_x = (d * e - b * f) / (a * d - b * c)
            CrossPoint_y = (a * f - c * e) / (a * d - b * c)
            #            print CrossPoint_x,CrossPoint_y
            if 0 <= CrossPoint_x <= len(ori) and 0 <= CrossPoint_y <= len(ori[0]):
                #            print mask[CrossPoint_x,CrossPoint_y,0]
                #                if not mask[CrossPoint_x,CrossPoint_y,0]==0:
                CrossPoints.append([CrossPoint_x, CrossPoint_y])

            cv2.circle(ori, (CrossPoint_x, CrossPoint_y), 40, (255, 255, 255), 1, lineType=4)
            cv2.circle(bubble, (CrossPoint_x, CrossPoint_y), 40, (255, 255, 255), 1, lineType=4)
    print ("CrossPoints:",len(CrossPoints))
    return len(CrossPoints),ori,bubble

count=0
vessel_Dir=os.path.dirname(os.getcwd())+'/VesselSegmentation/output/'
for src in os.listdir(vessel_Dir)[900:]:
#src="0000000000000623_rightresizemoveImageFinalnew.bmp"
    count=count+1
    src=src[0:src.find(".")]
    print( src,'(',count,'/',len(vessel_Dir),')')
    img = OpenPic(src,os.path.dirname(os.getcwd())+'/VesselSegmentation/output/')#cv2.imread('./input/vessel/'+src+".png")
    #img=Image.open('./input/vessel/'+src+".png")
    #img.show()
    kernel=np.uint8(np.zeros((5,5)))  
    for x in range(5):  
        kernel[x,2]=1;  
        kernel[2,x]=1;
    
    # if img is None:
    #     print "none!"
    #     a = np.zeros(shape=(584,564,3))
    #     for i in range(0, a.shape[0]):  # image.shape表示图像的尺寸和通道信息(高,宽,通道)
    #         for j in range(0, a.shape[1]):
    #             a[i, j] = 255
    #     cv2.imwrite("./output/"+src, a)
    #     cv2.imshow('',a)
    #     cv2.waitKey(0)
    #     continue
    # try:
    img = cv2.GaussianBlur(img,(5,5),0)
    ret,thresh=cv2.threshold(img,40,255,cv2.THRESH_BINARY)  #240
    edges = cv2.Canny(thresh, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,118)


    ori = OpenPic(src,os.path.dirname(os.getcwd())+'/VesselSegmentation/input/')#cv2.imread('./input/origin/'+src)
    _,mask=cv2.threshold(ori,40,255,cv2.THRESH_BINARY)
    value=0.0

    #经验参数
    default_threshold=45#20#55
    default_maxLineGap=5#10#2
    minLineLength = 200
    maxLineGap =default_maxLineGap
    threshold=default_threshold

    print ("threshold:", threshold)
    points,ori,bubble=DrawCircle(edges, threshold, minLineLength, maxLineGap)
    while points<800:
        if points < 500:
            maxLineGap-=1
        print("maxLineGap", maxLineGap)
        print ("less!!!")
        threshold-=1
        print ("threshold:",threshold)
        points, ori, bubble = DrawCircle(edges, threshold, minLineLength, maxLineGap)
    while points > 1400:
        print ("much!!!")
        if points > 5000:
            maxLineGap+=1
        threshold += 1
        print("maxLineGap",maxLineGap)
        print ("threshold:", threshold)
        points, ori, bubble = DrawCircle(edges, threshold, minLineLength, maxLineGap)
    threshold=default_threshold
    maxLineGap =default_maxLineGap
    print( "out\n")

    cv2.imwrite("combine.jpg",ori)
    cv2.imwrite("bubble.jpg",bubble)
    cv2.imwrite("./output/"+src+".jpg", bubble)
    img=cv2.resize(img,(500,500))
    bubble=cv2.resize(bubble,(500,500))
    mix=np.hstack([img,bubble])
    cv2.imwrite("./output/demo/"+src+".jpg", mix)
    cv2.destroyAllWindows()  
    # cv2.imshow(src,mix)
    # cv2.waitKey(200)
    # except:
    #     print "except"
    #     continue
print ("finished~~")
