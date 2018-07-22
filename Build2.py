import os
import numpy as np
import tensorflow as tf
import random
import shutil
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.draw import line
from binary_file_parser import *

pathname = "data/"
train_path = "train/"
test_path = "test/"
if os.path.isdir(train_path):
    shutil.rmtree(train_path)
os.makedirs(train_path)
if os.path.isdir(test_path):
    shutil.rmtree(test_path)
os.makedirs(test_path)
def render_image(drawing):
    img = np.zeros((256, 256), dtype=np.uint8)
    for tr in drawing:
        x,y = tr
        r0 = x[0]
        c0 = y[0]

        for i in range(1,len(x)):
            r1 = x[i]
            c1 = y[i]
            cc,rr = line(r0, c0, r1, c1)
            img[rr,cc] = 255
            r0 = r1
            c0 = c1

    image_resized = resize(img, (128, 128),anti_aliasing=False)
    
    return image_resized

def preprocess(source,classname,label):
    filename=source+classname+".bin"

    size = sum(1 for _ in unpack_drawings(filename))
    elements = random.sample(range(size), 1050)
    counter = 0
    elements.sort()
    #print(elements)
    for en in enumerate(unpack_drawings(filename)):
        i,drawing = en
        #print(i)
        if counter == 1050:
            break
        if elements[counter] == i:
            
            # do something with the drawing
            image_resized = render_image(drawing['image']).astype(np.int8).reshape(128*128).tostring()
    
            image_tfrecord = tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_resized]))
            label_tfrecord = tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))

            dict_tfrecord = {
                'image' : image_tfrecord,
                'label' : label_tfrecord
            }

            feature_tfrecord = tf.train.Features(feature = dict_tfrecord)
            example_tfrecord = tf.train.Example(features = feature_tfrecord)

            destiny = ("train/" if counter < 1000 else "test/") + classname + str(counter).zfill(4) + ".tfrecord"
            with tf.python_io.TFRecordWriter(destiny) as writer:
                writer.write(example_tfrecord.SerializeToString())

            counter+=1
            
    # random.shuffle(images)
    # #files_train = []
    # train = []
    # for i in range(0,1000):
    #     image = np.reshape(images[i],(1,128,128)).astype(bool)
    #     train.append(image)
    #   #name = classname+str(i).zfill(4)+".npy"
    #   #print(name)
    #   #files_train.append(path+name)
    #   #np.save(path+name,image
    # test = []
    # #files_test = []
    # for i in range(1000,1050):
    #     image = np.reshape(images[i],(1,128,128)).astype(bool)
    #     #name = classname+str(i).zfill(4)+".npy"
    #   #print(name)
    #     #files_test.append(path+name)
    #     #np.save(path+name,image)
    #     test.append(image)
    # #return files_train,files_test
    # return train,test


#print(preprocess("data/cat.bin","cat"))

#filenames_train = []
#filenames_test = []

with open("classes.txt", "r") as file:
    for en in enumerate(file.readlines()):
        k, s = en
        print("File: ", k)
        s = s[:-1]
        preprocess(pathname,s,k)

        #filenames_train.extend(ftrain)
        #filenames_test.extend(ftest)


#with open(train_path+"filenames.txt","w") as f:
#   f.write('\n'.join(filenames_train))


#with open(test_path+"filenames.txt","w") as f:
#   f.write('\n'.join(filenames_test))
