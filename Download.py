import wget
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
url = "http://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

with open("classes.txt","r") as file:
	for s in file.readlines():
		s = s[:-1]
		wget.download(url+s+".npy","data/"+s+".npy")
		