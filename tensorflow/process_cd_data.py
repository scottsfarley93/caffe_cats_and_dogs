import cv2

from skimage import color, io
from scipy.misc import imresize

import tflearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle, to_categorical
import numpy as np
from sklearn.cross_validation import train_test_split
from tflearn.metrics import Accuracy


img_prep = tflearn.ImagePreprocessing()

IMAGE_DIR = "./../smalltrain/"
p = 0
IMG_SIZE = 64
print "Imported all modules..."
def load_img(filename, do_resize=True, do_save=False):
    _file = os.path.split(filename)[1]
    imx = x = io.imread(filename)
    if do_resize:
        x = imresize(imx, (IMG_SIZE, IMG_SIZE, 3))
    if do_save:
        io.imsave( "./../smalltrain/" + _file,x)
    if _file[0:3] == "cat":
        label = [1, 0]
    elif _file[0:3] == "dog":
        label = [0, 1]
    return (x, label)

n = 25000

def load_dir(dir=IMAGE_DIR):
    X = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype='float64')
    Y = np.zeros((n, 2))
    idx = 0
    for _file in os.listdir(dir)[0:n]:
        filename = os.path.join(dir, _file)
        r = random.random()
        if r > p:
            x, y = load_img(filename, do_resize=False, do_save=False)
            if x is not None:
                # x = resize_img_array(x)
                X[idx] = np.array(x)
                Y[idx] = y
        idx += 1
        if (idx % 1000 == 0):
            print "Processed " + str(idx)
    return (X, Y)




X, Y = load_dir()
X, test_X, Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=42)
# Y = to_categorical(Y, 2)
# Y_test = to_categorical(test_Y, 2)


# Zero Center (With mean computed over the whole dataset)
img_prep.add_featurewise_zero_center()
# STD Normalization (With std computed over the whole dataset)
img_prep.add_featurewise_stdnorm()

print "defining network"

# Add these methods into an 'input_data' layer
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3],
                 data_preprocessing=img_prep, name='input')

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_2, 2)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_3 = conv_2d(network, 128, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_4 = conv_2d(network, 128, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_4, 2)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_5 = conv_2d(network, 128, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_5, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
# model = tflearn.DNN(network)

model = tflearn.DNN(network, tensorboard_verbose=0)


print "Training network"
# if os.path.exists("~/Documents/caffe_cats_and_dogs/tensorflow/model.model.meta"):
model.load("model.model")
print "Model loaded!"


# Training

model.fit(X, Y, validation_set=(test_X, test_Y),
      n_epoch=12, show_metric=True)

model.save("model.model")
