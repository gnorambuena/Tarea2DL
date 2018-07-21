import numpy as np
from matplotlib import pyplot as plt
from random import randint

lab = open("classes.txt","r").readlines()

A = np.load("test/images.npy")
L = np.load("test/labels.npy")

i = randint(0,4999)
img = A[i].astype(np.uint8)

print(lab[L[i]])
plt.imshow(img, cmap = 'gray_r')
plt.show()