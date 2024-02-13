#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:57:45 2017

@author: customer
"""

#import numpy as np
#import ConfigParser
#
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,normalization,concatenate

#U-Net
def get_unet(n_ch,num_lesion_class,patch_height,patch_width):
    inputs = Input((n_ch,patch_height, patch_width))
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
    #conv1 = Dropout(0.3)(conv1)
    #conv1 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv1)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    #pool1 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(pool1)
    conv2 = Convolution2D(64, (3, 3), padding='same')(pool1) #,activation='relu', padding='same')(pool1)    
    conv2 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv2)
    conv2 = core.Activation('relu')(conv2)
    #conv2 = Dropout(0.3)(conv2)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    #pool2 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(pool2)
    conv3 = Convolution2D(128, (3, 3), padding='same')(pool2)   #, activation='relu', padding='same')(pool2)    
   
    conv3 = normalization.BatchNormalization(epsilon=2e-05, axis=1, momentum=0.9, weights=None, beta_initializer='zero', gamma_initializer='one')(conv3)
    conv3 = core.Activation('relu')(conv3)
    #conv3 = Dropout(0.3)(conv3)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)#,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv3)
    

    #up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=1)
    conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up1)
    #conv4 = Dropout(0.3)(conv4)
    conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #conv4 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv4)
    #
    #up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=1)
    conv5 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up2)
    #conv5 = Dropout(0.3)(conv5)
    conv5 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv5)
    
    conv6 = Convolution2D(num_lesion_class+1, (1, 1), activation='relu',padding='same')(conv5)
    #conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)
    conv6 = core.Reshape(((num_lesion_class+1),patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)
    
    model = Model(inputs=inputs, outputs=conv7)
    return model
