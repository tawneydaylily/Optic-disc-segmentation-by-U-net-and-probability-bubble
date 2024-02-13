#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:53:42 2017

@author: huangyj
"""
import numpy as np
import matplotlib.pyplot as pl
import ConfigParser
import random
import sys

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,normalization
from keras.optimizers import Adam,Adagrad
from keras.metrics import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from keras.models import model_from_json
from NetModel import get_unet

from PIL import Image
from PIL import ImageFilter

K.set_image_dim_ordering('th')
#sys.path.append('')
sys.path.append('./lib/')
#import lib codes
from help_functions import *
from extract_patches import get_data_training
from pre_processing import *
#Config reader
config = ConfigParser.RawConfigParser()
config.read('./configuration.txt')
#data attributes
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
num_lesion_class = int(config.get('data attributes', 'num_lesion_class'))
N_subimgs = int(config.get('training settings', 'N_subimgs'))
Imgs_to_train = int(config.get('training settings', 'full_images_to_train'))
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
#paths
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
filename='./DRIVE/training/train'
path_data = config.get('data paths', 'path_local')
DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original')
DRIVE_train_groundTruth = path_data + config.get('data paths', 'train_groundTruth')
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
DRIVE_test_groundTruth = path_data + config.get('data paths', 'test_groundTruth')
#Experimental Status
NewTrain=True
N_ch=1
best_last = config.get('testing settings', 'best_last')

def CenterSampler(img_h,img_w,patch_h,patch_w,class_weight,mlist,Nimgs):
    class_weight=class_weight/np.sum(class_weight)
    p = random.uniform(0,1)
    psum=0
    label=0
    for i in range(class_weight.shape[0]):
        psum=psum+class_weight[i]
        if p<psum:
            label=i
            break
    if label==class_weight.shape[0]-1:
        i_center = random.randint(0,Nimgs-1)
        x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
         # print "x_center " +str(x_center)
        y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2)) 
    else:
        t=mlist[label]
        cid=random.randint(0,t[0].shape[0]-1)
        i_center=t[0][cid]
        y_center=t[1][cid]+random.randint(0-int(patch_w/2),0+int(patch_w/2))
        x_center=t[2][cid]+random.randint(0-int(patch_w/2),0+int(patch_w/2))
        #mask_shape=train_masks.shape[3]
        
        if y_center<patch_w/2:
            y_center=patch_w/2
        elif y_center>img_h-patch_w/2:
            y_center=img_h-patch_w/2
            
        if x_center<patch_w/2:
            x_center=patch_w/2
        elif x_center>img_w-patch_w/2:
            x_center=img_w-patch_w/2    
        
    return i_center,x_center,y_center
    
    

def Active_Generate(train_imgs,train_masks,patch_h,patch_w,batch_size,N_subimgs,N_imgs,class_weight,mlist):  
    while 1:
        img_h=train_imgs.shape[2]
        img_w=train_imgs.shape[3]
        for t in range(N_subimgs*N_imgs/batch_size):
            X=np.zeros([batch_size,N_ch,patch_h,patch_w])
            Y=np.zeros([batch_size,patch_h*patch_w,num_lesion_class+1])
            for j in range(batch_size):
                [i_center,x_center,y_center]=CenterSampler(img_h,img_w,patch_h,patch_w,class_weight,mlist,N_imgs)
                patch = train_imgs[i_center,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
                patch_mask = train_masks[i_center,:,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
                #X[j,:,:,:]=patch[1:2,:,:]
                img=patch
                img=Image.fromarray(((img*255).astype(np.uint8)).transpose(1,2,0))
                img=img.filter(ImageFilter.GaussianBlur(radius=random.randint(0,3)))
                X[j,0]=(np.asarray(img).astype(np.float32)/255.0).transpose(2,0,1)[1]
                Y[j,:,:]=masks_Unet(np.reshape(patch_mask,[1,num_lesion_class,patch_h,patch_w]),num_lesion_class)
            #else:
            yield (X, Y) 
            
def SampleTest(gen,batch_size,patch_h,patch_w):
    (X,Y)=next(gen)
    for i in range(batch_size):
        pl.imshow(X[i].transpose(1,2,0))
        pl.show()
        pl.imshow(np.reshape(Y,[batch_size,patch_h,patch_w,num_lesion_class+1])[i,:,:,0],cmap='gray')
        pl.show()

    


train_imgs_original = load_hdf5(DRIVE_train_imgs_original)#[img_id:img_id+1]
train_masks=np.zeros([Imgs_to_train,num_lesion_class,train_imgs_original.shape[2],train_imgs_original.shape[3]])
train_masks = load_hdf5(DRIVE_train_groundTruth +'.hdf5')#masks always the same

train_masks[:,0,:,:]=train_masks[:,0,:,:]
mlist=[np.where(train_masks[:,0,:,:]==np.max(train_masks[:,0,:,:]))]


test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
test_masks=np.zeros([Imgs_to_test,num_lesion_class,train_imgs_original.shape[2],train_imgs_original.shape[3]])
test_masks = load_hdf5(DRIVE_test_groundTruth +'.hdf5')#masks always the same



# visualize(group_images(train_imgs_original[0:20,:,:,:],5),'imgs_train')#.show()  #check original imgs train

#imgs=np.concatenate((train_imgs_original,test_imgs_original),axis=0)
#imgs=my_PreProc(imgs)
#train_imgs=imgs[0:N_imgs,:,:,:]
#test_imgs=imgs[N_imgs:N_imgs+Imgs_to_test,:,:,:]

train_imgs = my_PreProc(train_imgs_original)
#train_imgs = train_imgs_original
train_masks = train_masks/255.
train_imgs = train_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
train_masks = train_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565

test_imgs = my_PreProc(test_imgs_original)
#test_imgs = test_imgs_original
test_masks = test_masks/255.
test_imgs = test_imgs[:,:,9:574,:]  #cut bottom and top so now it is 565*565
test_masks = test_masks[:,:,9:574,:]  #cut bottom and top so now it is 565*565

#gen=generate_arrays_from_file(train_imgs,train_masks,patch_height,patch_width,batch_size,N_subimgs,N_imgs)
class_weight=np.array([0.0,1.0])#,60.0],20.0,90.0])#[10.0,30.0,20.0,60.0])#[10.0,30.0,20.0,60.0]
class_weight=class_weight/np.sum(class_weight)
test_class_weight=np.array([0.0,1.0])#,0.0,0.0,1.0])


mlist=[np.where(train_masks[:,0,:,:]==np.max(train_masks[:,0,:,:]))]
if np.max(train_masks[:,0,:,:])>1.0:
    for i in range(len(mlist[0][0])):
        train_masks[mlist[0][0][i],0,mlist[0][1][i],mlist[0][2][i]]=1.0
    mlist=[np.where(train_masks[:,0,:,:]==np.max(train_masks[:,0,:,:]))]

gen=Active_Generate(train_imgs,train_masks,patch_height,patch_width,batch_size,N_subimgs,Imgs_to_train,class_weight,mlist)
#SampleTest(gen,batch_size,patch_height,patch_width)

#one time test_sampling
test_mlist=[np.where(test_masks[:,0,:,:]==np.max(test_masks[:,0,:,:]))]
if np.max(test_masks[:,0,:,:])>1.0:    
    for i in range(len(mlist[0][0])):
        test_masks[test_mlist[0][0][i],0,test_mlist[0][1][i],test_mlist[0][2][i]]=1.0
    test_mlist=[np.where(test_masks[:,0,:,:]==np.max(test_masks[:,0,:,:]))]
    
test_gen=Active_Generate(test_imgs,test_masks,patch_height,patch_width,batch_size,N_subimgs,Imgs_to_test,test_class_weight,test_mlist)
#(test_X,test_Y)=next(test_gen)
#SampleTest(test_gen,batch_size,patch_height,patch_width)

model = get_unet(N_ch,num_lesion_class, patch_height, patch_width)

if NewTrain==False:
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
#plot_model(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)

checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', 
                               verbose=1, 
                               monitor='val_loss',
                               mode='auto', 
                               save_best_only=True) #save at each epoch if the validation decreased
hist=model.fit_generator(gen,
                    epochs=N_epochs, 
                    steps_per_epoch=N_subimgs*Imgs_to_train/batch_size,
                    verbose=1, 
                    callbacks=[checkpointer],
                    validation_data=test_gen,
                    validation_steps=N_subimgs*Imgs_to_test/batch_size)

with open('loss_plot.txt','w') as f:
    f.write(str(hist.history))
    
#model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)           

#GPU monitoring per 0.5s : watch -n 0.5 nvidia-smi

     