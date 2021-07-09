#!/usr/bin/env python
from glob import glob
from random import shuffle
import os
import cv2
import numpy as np
import h5py
import sys

DATADIR = "/data2/Naresh/data/BACH/final_train_test_val"

DATA_FILE = '/data2/Naresh/data/BACH/images_hdf5_new/train.h5'
train_addrs = glob(DATADIR+"/train/*/*jpg")
IMG_SIZE = 256
IMG_CHANNELS=3
train_shape = (len(train_addrs), IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
hdf5_file = h5py.File(DATA_FILE, mode='w')
hdf5_file.create_dataset('image', train_shape, np.uint8)#, compression="gzip")
hdf5_file.create_dataset('label', (len(train_addrs),), np.uint8 )
# Read images and save them
# Train images
for i in range(len(train_addrs)):
    if i%200 == 0 and i > 1 :
        print(f"Train: Done {i} of {len(train_addrs)}")
    
    addr = train_addrs[i]
    label = int(os.path.basename(os.path.dirname(addr)))
    #print(addr,label)
    #sys.exit(0)
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hdf5_file['image'][i, ...] = img
    hdf5_file['label'][i] = label
hdf5_file.close()

DATA_FILE = '/data2/Naresh/data/BACH/images_hdf5_new/val.h5'
test_addrs = glob(DATADIR+"/val/*/*jpg")
IMG_SIZE = 256
IMG_CHANNELS=3
test_shape = (len(test_addrs), IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
hdf5_file = h5py.File(DATA_FILE, mode='w')
hdf5_file.create_dataset('image', test_shape, np.uint8)#, compression="gzip")
hdf5_file.create_dataset('label', (len(test_addrs),), np.uint8 )
# Read images and save them
# Test images
for i in range(len(test_addrs)):
    if i%200 == 0 and i > 1 :
        print(f"Test: Done {i} of {len(test_addrs)}")
    addr = train_addrs[i]
    label = int(os.path.basename(os.path.dirname(addr)))
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hdf5_file['image'][i, ...] = img
    hdf5_file['label'][i] = label  
hdf5_file.close()

