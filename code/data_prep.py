import os
import glob
import random
import numpy as np
import cv2
import sys
sys.path.insert(1, "~/caffe/python")


import caffe
from caffe.proto import caffe_pb2
import lmbd





IMAGE_WIDTH  = 227
IMAGE_HEIGHT = 227
