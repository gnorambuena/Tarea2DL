import tensorflow as tf 
import numpy as np 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

train_data = np.load("traindata.npy")
train_label = np.load("trainlabel.npy")
test_data = np.load("testdata.npy")
test_label = np.load("testlabel.npy")

