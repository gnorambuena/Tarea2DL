import wget
import os
import shutil

url = "http://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
path = "data"
if os.path.isdir("/" + path):
    shutil.rmtree("/" + path)
os.makedirs(path)

with open("classes.txt", "r") as file:
    for en in enumerate(file.readlines()):
        i, s = en
        print("Downloading file: ",i)
        s = s[:-1]
        wget.download(url + s + ".npy", path + "/" + s + ".npy")
