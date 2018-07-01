import numpy as np

images = np.load("train/images.npy")
print(images.shape)
images = np.load("test/images.npy")
print(images.shape)