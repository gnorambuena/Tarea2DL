import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

url = "http://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
path = "data/"

Xtrain = []
Ytrain = []
Xtest = []
Ytest = []

with open("classes.txt", "r") as file:
    for en in enumerate(file.readlines()):
        k, s = en
        print("File: ", k)
        s = s[:-1]
        data = np.load(path + s + ".npy")
        data = data[np.random.choice(data.shape[0], 1050, replace=False)]

        data = np.array([np.reshape(t, (28, 28)) for t in data])
        # data = np.array([resize(t,(128,128),anti_aliasing = True) for t in data])

        traindata = data[:1000]
        Xtrain.append(traindata)
        Ytrain.append(np.array([k] * 1000))

        testdata = data[1000:]
        Xtest.append(testdata)
        Ytest.append(np.array([k] * 50))

Xtrain = np.concatenate(Xtrain).astype(float)
Xtrain = Xtrain/255
Ytrain = np.concatenate(Ytrain)

Xtest = np.concatenate(Xtest).astype(float)
Xtest = Xtest/255
Ytest = np.concatenate(Ytest)
np.save("traindata.npy", Xtrain)
np.save("trainlabel.npy", Ytrain)

np.save("testdata.npy", Xtest)
np.save("testlabel.npy", Ytest)

plt.imshow(Xtrain[1995], cmap='gray_r')
plt.show()
