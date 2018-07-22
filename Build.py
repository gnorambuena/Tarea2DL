import os
import numpy as np
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
            img[rr,cc] = 1
            r0 = r1
            c0 = c1
    image_resized = resize(img, (128, 128),anti_aliasing=False)
    
    for i in range(0,128):
        for j in range(0,128):
            if image_resized[i,j] != 0:
                image_resized[i,j] = 1
    return image_resized

def preprocess(path,classname):
    filename=path+classname+".bin"
    images = []
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
       		counter+=1
	        # do something with the drawing
	        image_resized = render_image(drawing['image'])
	        images.append(image_resized)
    random.shuffle(images)
    #files_train = []
    train = []
    for i in range(0,1000):
        image = np.reshape(images[i],(1,128,128)).astype(bool)
        train.append(image)
    	#name = classname+str(i).zfill(4)+".npy"
    	#print(name)
    	#files_train.append(path+name)
    	#np.save(path+name,image
    test = []
    #files_test = []
    for i in range(1000,1050):
        image = np.reshape(images[i],(1,128,128)).astype(bool)
        #name = classname+str(i).zfill(4)+".npy"
    	#print(name)
        #files_test.append(path+name)
        #np.save(path+name,image)
        test.append(image)
    #return files_train,files_test
    return train,test


#print(preprocess("data/cat.bin","cat"))

#filenames_train = []
#filenames_test = []
train = []
test = []
labels_train = []
labels_test = []
with open("classes.txt", "r") as file:
    for en in enumerate(file.readlines()):
        k, s = en
        if k == 10:
            break
        print("File: ", k)
        s = s[:-1]
        #ftrain,ftest = preprocess(pathname,s)

        #filenames_train.extend(ftrain)
        #filenames_test.extend(ftest)
        ftrain,ftest = preprocess(pathname,s)
        train.extend(ftrain)
        test.extend(ftest)
        labels_train.extend([k]*1000)
        labels_test.extend([k]*50)

train = np.concatenate(train)
test = np.concatenate(test)
#print(labels)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
print(train.shape)
print(test.shape)
print(labels_train.shape)
print(labels_test.shape)
np.save(train_path+"images.npy",train)
np.save(train_path+"labels.npy",labels_train)

np.save(test_path+"images.npy",test)
np.save(test_path+"labels.npy",labels_test)

#with open(train_path+"filenames.txt","w") as f:
#	f.write('\n'.join(filenames_train))


#with open(test_path+"filenames.txt","w") as f:
#	f.write('\n'.join(filenames_test))
