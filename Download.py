import wget
import os
import shutil

url = "https://storage.googleapis.com/quickdraw_dataset/full/binary/"
path = "data/"
if os.path.isdir(path):
    shutil.rmtree(path)
os.makedirs(path)

with open("classes.txt", "r") as file:
    for en in enumerate(file.readlines()):
        i, s = en
        print("\nDownloading file",i,": ", s)
        print("URL: ",url + s[:-1] + ".bin")
        s = s[:-1]
        wget.download(url + s + ".bin", path + s + ".bin")
