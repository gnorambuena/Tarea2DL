import os
import numpy as np
import random
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.draw import line
from binary_file_parser import *
import gc
pathname = "data/"

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
    files = []
    for i in range(0,1050):
    	image = images[0]
    	name = classname+str(i).zfill(4)+".npy"
    	#print(name)
    	files.append(path+name)
    	np.save(path+name,image)
    return files
#Xtrain = []
#Ytrain = []
#Xtest = []
#Ytest = []

#print(preprocess("data/cat.bin","cat"))

print("ok!")

filenames = []
labels = []
with open("classes.txt", "r") as file:
    for en in enumerate(file.readlines()):
        k, s = en
        print("File: ", k)
        s = s[:-1]
        filenames.extend(preprocess(pathname,s))
        #data = np.load(path + s + ".npy")
        #data = data[np.random.choice(data.shape[0], 1050, replace=False)]

        #data = np.array([np.reshape(t, (28, 28)) for t in data])
        # data = np.array([resize(t,(128,128),anti_aliasing = True) for t in data])

        #traindata = data[:1000]
        #Xtrain.append(traindata)
        labels.extend([k]*1000)
        #Ytrain.append(np.array([k] * 1000))

        #testdata = data[1000:]
        #Xtest.append(testdata)
        #Ytest.append(np.array([k] * 50))
#print(labels)
labels = np.array(labels)
np.save("labels.npy",labels)
with open("filenames.txt","w") as f:
	f.write('\n'.join(filenames))
#Xtrain = np.concatenate(Xtrain).astype(float)
#Xtrain = Xtrain/255
#Ytrain = np.concatenate(Ytrain)

#Xtest = np.concatenate(Xtest).astype(float)
#Xtest = Xtest/255
#Ytest = np.concatenate(Ytest)
#np.save("traindata.npy", Xtrain)
#np.save("trainlabel.npy", Ytrain)

#np.save("testdata.npy", Xtest)
#np.save("testlabel.npy", Ytest)

#plt.imshow(Xtrain[1995], cmap='gray_r')
#plt.show()
