# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from PIL import Image
import csv

ori_Dir =  os.listdir(os.path.dirname(os.getcwd())+'/VesselSegmentation/input/')
focus_Dir = os.listdir('./input/focus/')

num=0

"""
打开图片
"""
def OpenPic(src,Dir_str,is_focus=False):
    src_found=""
    Dir = os.listdir(Dir_str)
    if is_focus == True:
        for i in range(len(Dir)):
            if Dir[i].find(src)>=0 and Dir[i].find(".png")>0:
    #            print src[:len(src)-8],"*",focus_Dir[i]
                src_found=Dir[i]
                break
            else:
                if i is range(len(Dir)-1):
                    print (src,"PICTURE NOT FOUND!")
    else:
        for i in range(len(Dir)):
            if Dir[i].find(src)>=0:# and Dir[i].find(".png")>0:
    #            print src[:len(src)-8],"*",focus_Dir[i]
                src_found=Dir[i]
                break
            else:
                if i is range(len(Dir)-1):
                    print( src,"PICTURE NOT FOUND!")

    print( src_found)
    return cv2.imread(Dir_str+src_found)

"""
高斯滤波核
"""
kernel=np.uint8(np.zeros((5,5)))
for x in range(5):
    kernel[x,2]=1;
    kernel[2,x]=1;
    
def treat(img):
    kernel = np.uint8(np.zeros((3, 3)))
    for x in range(3):
        kernel[x, 2] = 1;
        kernel[2, x] = 1;
    ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)  # 240
    img = cv2.GaussianBlur(img, (9, 9), 0)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img = cv2.dilate(img, kernel)
    ret, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    ret, img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, kernel)  #################
    img = cv2.GaussianBlur(img, (13, 13), 0)
    ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, kernel)
    img = cv2.GaussianBlur(img, (13, 13), 0)
    img = cv2.erode(img, kernel)
    ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, (13, 13), 0)
    img = cv2.erode(img, kernel)
    ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    return img
def treat2(img):
    kernel=np.uint8(np.zeros((3,3)))  
    for x in range(3):  
        kernel[x,2]=1;  
        kernel[2,x]=1; 
    ret,img=cv2.threshold(img,180,255,cv2.THRESH_BINARY)  #240
    img = cv2.GaussianBlur(img,(9,9),0)
    ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY)
    img=cv2.dilate(img,kernel)
    ret,img=cv2.threshold(img,40,255,cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img,(5,5),0)
    ret,img=cv2.threshold(img,40,255,cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img,(9,9),0)
    ret,img=cv2.threshold(img,40,255,cv2.THRESH_BINARY)
    img = cv2.erode(img, kernel)#################
    img = cv2.GaussianBlur(img,(13,13),0)
    ret,img=cv2.threshold(img,30,255,cv2.THRESH_BINARY)
    img=cv2.erode(img,kernel)
    img = cv2.GaussianBlur(img, (13, 13), 0)
    img=cv2.erode(img,kernel)
    ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, (13, 13), 0)
    img = cv2.erode(img, kernel)
    ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    img = cv2.dilate(img, kernel)
    img = cv2.dilate(img, kernel)
    img = cv2.dilate(img, kernel)
    img = cv2.GaussianBlur(img, (13, 13), 0)
    img = cv2.erode(img, kernel)
    img = cv2.erode(img, kernel)
    img = cv2.erode(img, kernel)
    ret, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    return img

def make_sole(img):
    derp, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # for i in range(len(contours)):
    #     for j in range(len(contours[i])):
    #         cv2.circle(img, (contours[i][j][0][0], contours[i][j][0][1]), 10, (0, 20, 0), 1, lineType=4)
    def max_src(contours):
        area = 0
        for i in range(len(contours)):
            # print cv2.contourArea(contours[i])
            if area < cv2.contourArea(contours[i]):
                area = cv2.contourArea(contours[i])
                mark = i
        return mark

    a = cv2.boundingRect(contours[max_src(contours)])[0]
    b = cv2.boundingRect(contours[max_src(contours)])[1]
    c = cv2.boundingRect(contours[max_src(contours)])[2]
    d = cv2.boundingRect(contours[max_src(contours)])[3]
#    print contours[max_src(contours)]#,contours[max_src(contours)][1]
    # cv2.rectangle(img,(cv2.boundingRect(contours[max_src(contours)])[0],cv2.boundingRect(contours[max_src(contours)])[1]),(cv2.boundingRect(contours[max_src(contours)])[0]+cv2.boundingRect(contours[max_src(contours)])[2],cv2.boundingRect(contours[max_src(contours)])[1]+cv2.boundingRect(contours[max_src(contours)])[3]),color=(255,255,0))
    sole = np.empty([len(img), len(img[0])])
    circle = ori.copy()
    count = 0
    x_add = 0
    y_add = 0
    for i in range(1, len(img)):
        for j in range(1, len(img[0])):
            if i in range(a, a + c) and j in range(b, b + d) and img[j][i] == 255:
                sole[j][i] = 255
                x_add += i
                y_add += j
                count += 1
    sole = sole[:, :, np.newaxis]
    sole = np.concatenate((sole, sole, sole), axis=2)
    x_aver = x_add / (count)
    y_aver = y_add / (count)

    return sole,x_aver,y_aver,count

def draw_circle(img,x_aver,y_aver,count):
    distance_add = 0
    for i in range(1, len(img)):
        for j in range(1, len(img[0])):
            if img[i][j][0] == 255:
                distance = ((j - x_aver) ** 2 + (i - y_aver) ** 2) ** 0.5
                # print distance
                distance_add += distance
                count += 1
    aver = distance_add / count
    radius = int(aver / 0.38)
    circle = ori.copy()
    cv2.circle(circle, (x_aver, y_aver), radius, color=(255, 180, 0), thickness=2)
#    cv2.imshow("circle", circle)
#    cv2.waitKey(20)
    return circle,radius
def open_focus():
    '''
    打开病灶信息
    '''
    focus_src=''
    for i in range(len(focus_Dir)):
        if focus_Dir[i].find(src[:src.find('.')])>=0 and focus_Dir[i].find(".png")>=0:
            print (src[:src.find('.')],"*",focus_Dir[i])
            focus_src=focus_Dir[i]
            break
        else:
            if i is range(len(focus_Dir)-1):
                print (src,"FOCUS NOT FOUND!")
    focus = cv2.imread("./input/focus/"+focus_src)[:,:,0]
    _, focus = cv2.threshold(focus, 10, 255, cv2.THRESH_BINARY)
    focus = cv2.dilate(focus, kernel)
    focus = cv2.dilate(focus, kernel)
    focus = cv2.dilate(focus, kernel)
    focus = cv2.dilate(focus, kernel)
    focus = cv2.dilate(focus, kernel)
    focus = cv2.dilate(focus, kernel)
    return focus

with open('./output/OD_result.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    spamwriter.writerow(["Name", "Point_x", "Point_y", "Radius"])
    for src in ori_Dir:
        print (src,num)
        ori=cv2.imread(os.path.dirname(os.getcwd())+'/VesselSegmentation/input/'+src)
    #    vessel=cv2.imread('./Vessel Aid/input/vessel/'+src+".png")
        vessel=Image.open(os.path.dirname(os.getcwd())+'/VesselSegmentation/output/'+src+'.png')
        unet=cv2.imread(os.path.dirname(os.getcwd())+'/OD-unet-master/output/'+src+".png")
        bubble=cv2.imread(os.path.dirname(os.getcwd())+'/Vessel Aid/output/'+src[:src.find('.')]+".jpg")
        focus=open_focus();
    #    print ori, vessel, unet, bubble
        vessel = vessel.resize([ori.shape[1],ori.shape[0]])#,Image.BILINEAR)
#        vessel.save(os.path.dirname(os.getcwd())+'/Vessel Aid/input/vessel/'+src)
#        vessel=cv2.imread(os.path.dirname(os.getcwd())+'/Vessel Aid/input/vessel/'+src)
#        mix = cv2.addWeighted(vessel, 0.5, ori, 1-0.5, 0)
    #    cv2.imshow("ori",ori)
    #    cv2.imshow("vessel",vessel)
    #    cv2.imshow("unet",unet)
    #    cv2.imshow("mix",mix)
    #    vessel.show()
    #    cv2.imshow("bubble",bubble)
    #     mix = np.concatenate((ori,vessel,unet,bubble), axis=1)
        overlay=unet.copy()#把视盘概率、病灶、泡泡信息叠加
        overlay_demo=overlay.copy()
        focus = cv2.resize(focus,(len(bubble[0]),len(bubble)))
        for i in range(len(overlay)):
            for j in range(len(overlay[0])):
                if bubble[i][j][0]==0 and focus[i][j]>0:
                    overlay[i][j][0]=0
                    overlay[i][j][1]=0
                    overlay[i][j][2]=0
                    overlay_demo[i][j][0]=0
                    overlay_demo[i][j][1]=0
                    overlay_demo[i][j][2]=focus[i][j]
                elif not bubble[i][j][0]==0:
                    overlay_demo[i][j][0]=255
        
#        for i in range(len(overlay_demo)):
#            for j in range(len(overlay_demo[0])):
#                if focus[i][j]>10:
#                    overlay_demo[i][j][2]=focus[i][j]
#        cv2.imshow("overlay_demo",overlay_demo)
#        cv2.waitKey(20)  
        
#        many=treat2(overlay)
        cv2.imwrite('./output/many/' + src, treat2(overlay))
        many=cv2.imread("./output/many/"+src,0)
        for i in range(len(overlay)):
            for j in range(len(overlay[0])):
                if focus[i][j]>0:
                    many[i][j]=0
        sole, xx, yy, count=make_sole(many)
        print ("center: (",xx,",",yy,")")
        estimate_pic,radius=draw_circle(sole, xx, yy, count)
        print ("radius: ",radius)
        cv2.imwrite('./output/sole/' + src, sole)
        cv2.imwrite('./output/result/'+src,estimate_pic)
        
        estimate_pic=cv2.resize(estimate_pic,(500,500))
        many = many[:, :, np.newaxis]
        many = np.concatenate((many, many, many), axis=2)
        mix = np.hstack([overlay_demo,treat2(overlay), estimate_pic])
        cv2.imwrite('./output/demo/'+src,mix)
#        cv2.imshow("mix",mix)
#        cv2.waitKey(0) 
        
        spamwriter.writerow([src,xx, yy,radius])
        # sole=cv2.imread('./output/sole/' + src)

        # mix1 = np.concatenate((ori,unet, bubble), axis=1)
        # mix2 = np.concatenate((overlay,treat2(overlay),sole), axis=1)
        #
        # mix = np.concatenate((mix1,mix2), axis=0)
        cv2.destroyAllWindows()
        # cv2.imshow(src,mix)
        # cv2.waitKey(1000)
        num+=1
csvfile.close()

watchout=[]
lines = []
with open('./output/OD_result.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        lines.append(row)
    for i in range(1,len(lines)):
        lines[i-1]=lines[i]
    lines[i]=[]
    x_add=0
    y_add=0
    r_add=0
    for i in range(len(lines)-1):
        x_add+=int(lines[i][1])
        y_add+=int(lines[i][2])
        r_add+=int(lines[i][3])
    x_aver=x_add/len(lines)
    y_aver=y_add/len(lines)
    r_aver=r_add/len(lines)

    print ("Watch out these files:")
    for i in range(len(lines)-1):
        if (int(lines[i][1])-x_aver)**2>300 or (int(lines[i][2])-y_aver)**2>300 or (int(lines[i][3])-r_aver)**2>250:
            print (lines[i][0])
            watchout.append(lines[i][0])
csvfile.close()

with open('./output/OD_result.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel')
    spamwriter.writerow(["Name", "Point_x", "Point_y", "Radius"])
    for i in range(len(lines)-1):
        if lines[i][0] in watchout:
            spamwriter.writerow([lines[i][0], lines[i][1], lines[i][2], lines[i][3],"Warning"])
        else:
            spamwriter.writerow([lines[i][0], lines[i][1], lines[i][2], lines[i][3]])
csvfile.close()

print ("finished~~")
cv2.destroyAllWindows()