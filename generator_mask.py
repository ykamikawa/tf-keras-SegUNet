# -*- coding: utf-8 -*-
import numpy as np
import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize


def msklab(labels, dims, target_num):
    x = np.zeros([dims[0], dims[1],1])
    for i in range(dims[0]):
        for j in range(dims[1]):
            if(labels[i, j, 0] == target_num):
                x[i, j, 0] = 1
    x = x.reshape(dims[0] * dims[1], 1)
    return x


# generator that we will use to read the data from the directory
def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
    while True:
        ix = np.random.choice(np.arange(len(lists)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            original_img = cv2.imread(img_dir + lists.iloc[i, 0]+".jpg")[:, :, ::-1]
            resized_img = cv2.resize(original_img, (dims[0], dims[1]))
            array_img = img_to_array(resized_img)/255
            imgs.append(array_img)
            # masks
            original_mask = cv2.imread(mask_dir + lists.iloc[i, 0] + '.png')
            resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
            array_mask = msklab(resized_mask, dims, 9)
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels
