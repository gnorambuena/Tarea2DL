import tensorflow as tf 
import numpy as np 
import os
from Model import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_n = 100000
train_data = np.load("traindata.npy").astype(float)
train_data = np.reshape(train_data,(train_n,img_size,img_size,channels))
train_label = np.load("trainlabel.npy").astype(int)

test_n = 5000
test_data = np.load("testdata.npy").astype(float)
test_data = np.reshape(test_data,(test_n,img_size,img_size,channels))
test_label = np.load("testlabel.npy").astype(int)

batch_size = 1
epochs = 50
dataset = tf.data.Dataset.from_tensor_slices((train_data,train_label))
dataset = dataset.shuffle(buffer_size = train_n, reshuffle_each_iteration = True)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(epochs)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

init = tf.global_variables_initializer()

with tf.Session(config = config) as sess:
    sess.run(init)
    x,y_label = sess.run(next_element)
    sess.run(optimizer)
