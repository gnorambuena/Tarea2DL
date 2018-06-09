import wget
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

url = "http://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
path = "data/"
aux = "aux.npy"

try:
    os.remove(aux)
except:
    pass

Xtrain = np.array([])
Xtrain = np.reshape(Xtrain, (0, 28, 28))
Ytrain = np.array([])
Xtest = np.array([])
Xtest = np.reshape(Xtest, (0, 28, 28))
Ytest = np.array([])
with open("classes.txt", "r") as file:
    k = 0
    for s in file.readlines():
        print("File: ",k)
        s = s[:-1]
        data = np.load(path+s+".npy")
        data = data[np.random.choice(data.shape[0], 1050, replace=False)]

        data = np.array([np.reshape(t, (28, 28)) for t in data])
        # data = np.array([resize(t,(128,128),anti_aliasing = True) for t in data])

        traindata = data[:1000]
        Xtrain = np.concatenate((Xtrain, traindata))
        Ytrain = np.concatenate((Ytrain, np.array([k] * 1000)))

        testdata = data[1000:]
        Xtest = np.concatenate((Xtest, testdata))
        Ytest = np.concatenate((Ytest, np.array([k] * 50)))

        k += 1

np.save("traindata.npy", Xtrain)
np.save("trainlabel.npy", Ytrain)

np.save("testdata.npy", Xtest)
np.save("testlabel.npy", Ytest)

plt.imshow(Xtrain[1995], cmap='gray_r')
plt.show()
