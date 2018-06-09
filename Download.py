import wget

url = "http://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

with open("classes.txt", "r") as file:
    for s in file.readlines():
        s = s[:-1]
        wget.download(url + s + ".npy", "data/" + s + ".npy")
