import random

classes = []
with open("categories.txt", "r") as file:
    classes = file.readlines()
    classes = random.sample(classes, 100)

with open("classes.txt", "w") as file:
    file.writelines(classes)
