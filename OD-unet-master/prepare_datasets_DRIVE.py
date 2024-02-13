#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import cv2


def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./DRIVE/training/images/"
groundTruth_imgs_train = "./DRIVE/training/remove_vessel/"
borderMasks_imgs_train = "./DRIVE/training/mask/"
#test
original_imgs_test = "./DRIVE/test/images/"
groundTruth_imgs_test = "./DRIVE/test/1st_manual/"
borderMasks_imgs_test = "./DRIVE/test/mask/"
#---------------------------------------------------------------------------------------------

Nimgs = 95#20
channels = 3
height = 500#584
width = 500#565
dataset_path = "./DRIVE_datasets_training_testing/"

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    abnormal=[]
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):#len(files)
            #original
            print "original image: " +files[i]

            img = Image.open(imgs_dir+files[i])
            if img.size==(500,500):

                imgs[i] = np.asarray(img)
                #corresponding ground truth
                groundTruth_name = files[i]
                print "ground truth name: " + groundTruth_name
                # g_truth = Image.open(groundTruth_dir + groundTruth_name)
                g_truth = Image.open(imgs_dir + groundTruth_name)

                if train_test=="train":
                    groundTruth[i] = np.asarray(g_truth)[:, :, 0]
                elif train_test=="test":
                    groundTruth[i] = np.asarray(g_truth)[:, :, 0]

                for ii in range(len(groundTruth[i])):
                    for jj in range(len(groundTruth[i][0])):
                        if groundTruth[i][ii][jj] == 3:
                            groundTruth[i][ii][jj]=0
                        elif groundTruth[i][ii][jj] == 4:
                            groundTruth[i][ii][jj]=0
    #                    if groundTruth[i][ii][jj] == 255:
    #                        print "255255255255255255255255255#####"
    #                    print groundTruth[i][ii][jj]

                #corresponding border masks
                border_masks_name = ""
                if train_test=="train":
                    border_masks_name = files[i]
                elif train_test=="test":
                    border_masks_name = files[i]
                else:
                    print "specify if train or test!!"
                    exit()
                print "border masks name: " + border_masks_name
                img=cv2.imread(imgs_dir+files[i])
                ret,thresh1=cv2.threshold(img,1,255,cv2.THRESH_BINARY)
                b_mask = thresh1
                border_masks[i] = np.asarray(b_mask)[:,:,0]
            else:
                print "abnormal size:"+files[i]
                abnormal.append(files[i])
                print img.size

    print "imgs max: " +str(np.max(imgs))
    print "imgs min: " +str(np.min(imgs))
    # assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    # assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print "ground truth and border masks are correctly withih pixel value range 0-255 (black-white)"
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,"train")
print "saving train datasets"
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")

Nimgs = 4#20
channels = 3
height = 500#584
width = 500#565
#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,"test")
print "saving test datasets"
write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")
