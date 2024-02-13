#==========================================================
#
#  This prepare the hdf5 datasets of the Dataset database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./Dataset/training/images/"
groundTruth_imgs_train = "./Dataset/training/1st_manual/"
borderMasks_imgs_train = "./Dataset/training/mask/"
#test
original_imgs_test = "./Dataset/test/images/"
groundTruth_imgs_test = "./Dataset/test/1st_manual/"
borderMasks_imgs_test = "./Dataset/test/mask/"
#---------------------------------------------------------------------------------------------
import configparser
config = configparser.RawConfigParser()
config.read('configuration.txt')
dataset_path = "./HDF/"
Imgs_to_train = int(config.get('training settings', 'full_images_to_train'))
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
channels = 3
height = 500#584
width = 500#565

def get_datasets(imgs_dir,groundTruth_dir,N_imgs,train_test="null"):
    imgs = np.empty((N_imgs,height,width,channels))
    groundTruth= np.empty((N_imgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            img=img.resize([width,height],Image.BILINEAR)
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print("ground truth name: " + groundTruth_name)
            #Label interpolation should be NEAREST, otherwise the label will not be binary and ROC curve reports error.
            g_truth = Image.open(groundTruth_dir + groundTruth_name).resize([width,height],Image.NEAREST)#.convert('L')                                    
            groundTruth[i] = np.asarray(g_truth)


    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255)
    assert(np.min(groundTruth)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (N_imgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(N_imgs,1,height,width))
    assert(groundTruth.shape == (N_imgs,1,height,width))

    return imgs, groundTruth


#getting the training datasets
imgs_train, groundTruth_train = get_datasets(original_imgs_train,groundTruth_imgs_train,Imgs_to_train,"train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "groundTruth_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test = get_datasets(original_imgs_test,groundTruth_imgs_test,Imgs_to_test,"test")
print("saving test datasets")
write_hdf5(imgs_test,dataset_path + "imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "groundTruth_test.hdf5")
