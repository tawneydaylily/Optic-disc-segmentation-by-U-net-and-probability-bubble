#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 21:01:16 2017

@author: huangyj
"""
import sys
import importlib
importlib.reload(sys)
# sys.setdefaultencoding('gb18030')
import numpy as np
import configparser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as pl
import time
from PIL import Image
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# sys.path.append('./lib/')
# help_functions.py

from help_functions import *
from extract_patches import *
from pre_processing import *




# Load the original data and return the extracted patches for testing
# return the ground truth in its original shape
def get_data_testing_overlap(test_imgs_original, patch_height, patch_width, stride_height, stride_width):
    ### test   
    test_imgs = my_PreProc(test_imgs_original)
    print("\ntest images shape:")
    print(test_imgs.shape)


    #extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)

    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " +str(np.min(patches_imgs_test)) +' - '+str(np.max(patches_imgs_test)))

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3]




#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('./configuration.txt')
#===========================================
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')
#original test images (for FOV selection)

#test_imgs_orig = Image.open('./Input/054_test.bmp')#load_hdf5(DRIVE_test_imgs_original)
#test_imgs_orig = test_imgs_orig.resize([564,584],Image.BILINEAR)
#test_imgs_orig = np.asarray(test_imgs_orig)
#test_imgs_orig = np.reshape(test_imgs_orig,[1,584,564,3])
#test_imgs_orig = np.transpose(test_imgs_orig,(0,3,1,2))


full_img_height =500#test_imgs_orig.shape[2]
full_img_width = 500#test_imgs_orig.shape[3]
#the border masks provided by the DRIVE
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
num_lesion_class = int(config.get('data attributes', 'num_lesion_class'))
#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)
#model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
#N full images to be predicted
Imgs_to_test = 1
#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
#====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')
best_last = config.get('testing settings', 'best_last')
N_ch=3


model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
orig_img_g=np.zeros([Imgs_to_test,3,500,500])
pred_g=np.zeros([Imgs_to_test,3,500,500])#num_lesion_class
#============ Load the data and divide in patches
for filename in os.listdir('./Input/'):
    #for filename in filenames:
        test_imgs_orig = Image.open('./Input/'+filename)#load_hdf5(DRIVE_test_imgs_original)
        test_imgs_orig = test_imgs_orig.resize([500,500],Image.BILINEAR)
        test_imgs_orig = np.asarray(test_imgs_orig)
        test_imgs_orig = np.reshape(test_imgs_orig,[1,500,500,3])
        test_imgs_orig = np.transpose(test_imgs_orig,(0,3,1,2))
    
        time1= time.time()
        patches_imgs_test = None
        new_height = None
        new_width = None
        patches_masks_test = None
        if average_mode == True:
            patches_imgs_test, new_height, new_width = get_data_testing_overlap(
                test_imgs_orig,
                patch_height = patch_height,
                patch_width = patch_width,
                stride_height = stride_height,
                stride_width = stride_width
            )

        # print "!!!!!!", patches_imgs_test.shape
        
        #================ Run the prediction of the patches ==================================
        best_last = config.get('testing settings', 'best_last')
        #Load the saved model
    
        #Calculate the predictions
        print("!!!!!!!!!!!!!!!!!!!",patches_imgs_test[:,1:2].shape)
        #predictions = model.predict(patches_imgs_test.transpose([0,2,3,1]), batch_size=32, verbose=2)
        predictions = model.predict(patches_imgs_test[:,1:2], batch_size=32, verbose=2)
    #    for pid in range(11445):
    #        for pixel in range(48*48):
    #            loc=np.where(predictions[pid,pixel,:]==np.max(predictions[pid,pixel,:]))
    #            label=np.zeros([4])
    #            label[loc[0][0]]=1.0
    #            predictions[pid,pixel,:]=loc
    
        print("predicted images size :")
        print(predictions.shape)
        
        #===== Convert the prediction arrays in corresponding images
        #pred_patches = pred_to_imgs(predictions,"original")
        pred_patches = pred_to_imgs(predictions,"original")
        #if Tensorflow:
        pred_patches=pred_patches.transpose(0,3,1,2)
        
        #========== Elaborate and visualize the predicted images ====================
        pred_imgs = None
        orig_imgs = None
        gtruth_masks = None
        #if average_mode == True:
        print(pred_patches.shape,"~~")
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
        orig_imgs = my_PreProc(test_imgs_orig[:,:,:,:])    #originals
        #else:
        #    pred_imgs = recompone(pred_patches,13,12)       # predictions
        #    orig_imgs = recompone(patches_imgs_test,13,12)  # originals
        ## back to original dimensions
        orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
        pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
                    
        #pred_imgs=pred_imgs[:,0:3,:,:]
        pred_imgs=pred_imgs[:,0:1,:,:]
        pred_imgs=pred_imgs.repeat(3,axis=1)
        
        #gtruth_masks1[:,1,:,:]=gtruth_masks[:,1,:,:]
        #gtruth_masks1[:,2,:,:]=gtruth_masks[:,2,:,:]+gtruth_masks[:,3,:,:]
        #gtruth_masks1=gtruth_masks1+gtruth_masks[:,3:4,:,:]
        print("Orig imgs shape: " +str(orig_imgs.shape))
        print("pred imgs shape: " +str(pred_imgs.shape))
        #visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")#.show()
        #visualize(group_images(pred_imgs,N_visual),path_experiment+"all_predictions")#.show()
        #visualize(group_images(gtruth_masks1,N_visual),path_experiment+"all_groundTruths")#.show()
        #visualize results comparing mask and prediction:
        #assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
        N_predicted = orig_imgs.shape[0]
        group = N_visual
        assert (N_predicted%group==0)
        #for i in range(int(N_predicted/group)):
        i=0
        orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
        orig_stripe[:,:,0]=orig_stripe[:,:,1]
        orig_stripe[:,:,2]=orig_stripe[:,:,1]
        pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
        # total_img = np.concatenate((orig_stripe,pred_stripe),axis=0)
        # pl.imshow(total_img)
        visualize(pred_stripe,"./output/"+filename)#.show()
        time2= time.time()
        print('\n Time Used ',time2-time1, 'seconds\n')

