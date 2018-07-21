import os

for s in open("classes.txt","r").readlines():
	for k in range(1050):
		os.system("rm data/"+s[:-1] + str(k).zfill(4) + ".npy")