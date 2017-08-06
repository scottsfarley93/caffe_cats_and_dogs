import os
import glob
import random
import numpy as np
import cv2
import sys
sys.path.insert(1, "~/caffe/python")


import caffe
from caffe.proto import caffe_pb2

caffe.set_mode_cpu()

net = caffe.Net('./../caffe_models/conv.prototxt', caffe.TEST)

[(k, v.data.shape) for k, v in net.blobs.items()]
