import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle as pkl

from ResNet import ResNet

resnet = pkl.load(open("./Models/ResNet_animals10.pkl", "rb"))

class_dictionary = {'butterfly': 0, 'cat': 1, 'chicken':2 , 'cow': 3,
                    'dog': 4, 'elephant': 5, 'horse': 6, 'sheep': 7,
                    'spider': 8, 'squirrel': 9}
path = "./ResNet_1000/"

acc = 0
for image in os.listdir(path):

    path_image = path + image
    img = Image.open(path_image)
    img = np.asarray(img)
    img = img[:,:,:3]

    real_class = image.split("_")[0]
    real_class = class_dictionary[real_class]
    predicted_class = np.argmax(resnet.predict_numpy_images([img]))

    if real_class == predicted_class:
        print(image)
        acc = acc + 1

print(acc/len(os.listdir(path)))
