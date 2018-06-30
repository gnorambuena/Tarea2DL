import os
import numpy as np
import random
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.draw import line
from binary_file_parser import *

pathname = "data/"
train_path = "train/"
test_path = "test/"

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
    files_train = []
    for i in range(0,1000):
    	image = images[i]
    	name = classname+str(i).zfill(4)+".npy"
    	#print(name)
    	files_train.append(path+name)
    	np.save(path+name,image)
    files_test = []
    for i in range(1000,1050):
        image = images[i]
        name = classname+str(i).zfill(4)+".npy"
    	#print(name)
        files_test.append(path+name)
        np.save(path+name,image)

    return files_train,files_test


#print(preprocess("data/cat.bin","cat"))

filenames_train = []
filenames_test = []
labels_train = []
labels_test = []
with open("classes.txt", "r") as file:
    for en in enumerate(file.readlines()):
        k, s = en
        print("File: ", k)
        s = s[:-1]
        ftrain,ftest = preprocess(pathname,s)

        filenames_train.extend(ftrain)
        filenames_test.extend(ftest)
        labels_train.extend([k]*1000)
        labels_test.extend([k]*50)

#print(labels)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
np.save(train_path+"labels.npy",labels_train)
np.save(test_path+"labels.npy",labels_test)

with open(train_path+"filenames.txt","w") as f:
	f.write('\n'.join(filenames_train))


with open(test_path+"filenames.txt","w") as f:
	f.write('\n'.join(filenames_test))
