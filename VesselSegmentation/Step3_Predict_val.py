#!/usr/bin/env python2
print('hey')
import sys
import importlib
importlib.reload(sys)
sys.setdefaultencoding('gb18030')

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
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as pl
import time

sys.path.append('./lib/')
# help_functions.py
from help_functions import *
from extract_patches import *
from pre_processing import *



print('start')

#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('./configuration.txt')
#===========================================
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')
#original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
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
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
#====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')
best_last = config.get('testing settings', 'best_last')
N_ch=3
# #ground truth
# gtruth= path_data + config.get('data paths', 'test_groundTruth')
# img_truth= load_hdf5(gtruth)
# visualize(group_images(test_imgs_orig[0:Imgs_to_test,:,:,:],5),'original')#.show()
# visualize(group_images(test_border_masks[0:Imgs_to_test,:,:,:],5),'borders')#.show()
# visualize(group_images(img_truth[0:Imgs_to_test,:,:,:],5),'gtruth')#.show()

model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
orig_img_g=np.zeros([Imgs_to_test,3,584,565])
gt_g=np.zeros([Imgs_to_test,3,584,565])#num_lesion_class
pred_g=np.zeros([Imgs_to_test,3,584,565])#num_lesion_class
#============ Load the data and divide in patches
for img_id in range(Imgs_to_test):
    time1= time.time()
    patches_imgs_test = None
    new_height = None
    new_width = None
    masks_test  = None
    patches_masks_test = None
    if average_mode == True:
        patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
            DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
            DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
            Img_id = img_id,
            patch_height = patch_height,
            patch_width = patch_width,
            stride_height = stride_height,
            stride_width = stride_width
        )
    else:
        patches_imgs_test, patches_masks_test = get_data_testing(
            DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
            DRIVE_test_groudTruth = path_data + config.get('data paths', 'test_groundTruth'),  #masks
            Imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),
            patch_height = patch_height,
            patch_width = patch_width,
        )
    
    
    
    #================ Run the prediction of the patches ==================================
    best_last = config.get('testing settings', 'best_last')
    #Load the saved model

    #Calculate the predictions
    
    #predictions = model.predict(patches_imgs_test.transpose([0,2,3,1]), batch_size=32, verbose=2)
    predictions = model.predict(patches_imgs_test[:,1:2,:,:], batch_size=32, verbose=2)
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
    if average_mode == True:
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
        orig_imgs = my_PreProc(test_imgs_orig[img_id:img_id+1,:,:,:])    #originals
        gtruth_masks = masks_test  #ground truth masks
    else:
        pred_imgs = recompone(pred_patches,13,12)       # predictions
        orig_imgs = recompone(patches_imgs_test,13,12)  # originals
        gtruth_masks = recompone(patches_masks_test,13,12)  #masks
    # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
    #kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
    ## back to original dimensions
    orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
    pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
    
    hardclass=False
    if hardclass:
        for pid in range(1):
            for py in range(584):
                for px in range(565):
                    loc=np.where(pred_imgs[pid,:,py,px]==np.max(pred_imgs[pid,:,py,px]))
                    for channel in range(4):
                        if channel==loc[0][0]:
                            pred_imgs[pid,channel,py,px]=1.0
                        else:
                            pred_imgs[pid,channel,py,px]=0.0
            #label=np.zeros([4])
            #label[loc[0][0]]=1.0
            #pred_imgs[pid,:,py,px]=label
            
    #pred_imgs=pred_imgs[:,0:3,:,:]
    pred_imgs=pred_imgs[:,0:1,:,:]
    pred_imgs=pred_imgs.repeat(3,axis=1)
    
    gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
    gtruth_masks1=np.zeros([Imgs_to_test,3,584,565])
    gtruth_masks1[:,0,:,:]=gtruth_masks[:,0,:,:]
    gtruth_masks1[:,1,:,:]=gtruth_masks[:,0,:,:]
    gtruth_masks1[:,2,:,:]=gtruth_masks[:,0,:,:]
    #gtruth_masks1[:,1,:,:]=gtruth_masks[:,1,:,:]
    #gtruth_masks1[:,2,:,:]=gtruth_masks[:,2,:,:]+gtruth_masks[:,3,:,:]
    #gtruth_masks1=gtruth_masks1+gtruth_masks[:,3:4,:,:]
    print("Orig imgs shape: " +str(orig_imgs.shape))
    print("pred imgs shape: " +str(pred_imgs.shape))
    print("Gtruth imgs shape: " +str(gtruth_masks.shape))
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
    # orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    # masks_stripe = group_images(gtruth_masks1[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    # total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
    visualize(pred_stripe,path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(img_id))#.show()
    time2= time.time()
    print('\n Time Used ',time2-time1, 'seconds\n')
    orig_img_g[img_id,:,:,:]=orig_imgs[0]
    gt_g[img_id,:,:,:]=gtruth_masks1[0]#masks_test[0]
    pred_g[img_id,:,:,:]=pred_imgs[0]


           
    
#compute AUC/ROC
name_lesions=['Cotton','Exudate','Hemorr']
for channel in range(num_lesion_class):
    y_scores, y_true = pred_only_FOV(pred_g[:,channel:channel+1,:,:],gt_g[:,channel:channel+1,:,:])#, test_border_masks)
    #Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " +str(AUC_ROC))
    roc_curve_fig =plt.figure()
    plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment+name_lesions[channel]+"_"+"ROC.png")
    
    #Precision-recall curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision,recall)
    print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment+name_lesions[channel]+"_"+"Precision_recall.png")
    
    #Confusion matrix
    threshold_confusion = 0.4
    print("\nConfusion matrix:  Costum threshold (for positive) of " +str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i]>=threshold_confusion:
            y_pred[i]=1
        else:
            y_pred[i]=0
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print("Global Accuracy: " +str(accuracy))
    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print("Specificity: " +str(specificity))
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print("Sensitivity: " +str(sensitivity))
    precision = 0
    if float(confusion[1,1]+confusion[0,1])!=0:
        precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
    print("Precision: " +str(precision))
    
    #Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print("\nJaccard similarity score: " +str(jaccard_index))
    
    #F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " +str(F1_score))
    
    #Save the results
    file_perf = open(path_experiment+name_lesions[channel]+"_"+'performances.txt', 'w')
    file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                    + "\nJaccard similarity score: " +str(jaccard_index)
                    + "\nF1 score (F-measure): " +str(F1_score)
                    +"\n\nConfusion matrix:"
                    +str(confusion)
                    +"\nACCURACY: " +str(accuracy)
                    +"\nSENSITIVITY: " +str(sensitivity)
                    +"\nSPECIFICITY: " +str(specificity)
                    +"\nPRECISION: " +str(precision)
                    )
    file_perf.close()