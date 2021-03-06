import tensorflow as tf 
import numpy as np 
import os
from Model import *

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.4

train_n = 100000
train_data = np.load("traindata.npy").astype(float)
train_data = np.reshape(train_data,(train_n,img_size,img_size,channels))
train_label = np.load("trainlabel.npy").astype(int)
#train_label = np.array([[1 if j == train_label[i] else 0 for j in range(classes)] for i in range(train_n)]).astype(float)
train_label = tf.one_hot(train_label,100)

test_n = 5000
test_data = np.load("testdata.npy").astype(float)
test_data = np.reshape(test_data,(test_n,img_size,img_size,channels))
test_label = np.load("testlabel.npy").astype(float)

batch_size = 1
epochs = 50
dataset = tf.data.Dataset.from_tensor_slices((train_data,train_label))
dataset = dataset.shuffle(buffer_size = train_n, reshuffle_each_iteration = True)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(epochs)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(init)
    x,y_label = sess.run(next_element)
    sess.run(optimizer)
