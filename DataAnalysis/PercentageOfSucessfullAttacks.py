import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle as pkl
import cv2 as cv

# Needed for pickle loading!!!
from ResNet import ResNet
from VGG import VGG
from GoogLeNet import GoogLeNet
from AlexNet import AlexNet

# Parameters ######################################################

filter = False
model = "GoogleNet"
image_folder = "D:/Captcha_Data/ResNet_1000_3ModelClean/"
ResNet_path = "C:/Users/33782/Desktop/Animals10Captcha/Models/ResNet_animals10.pkl"
VGG_path = "C:/Users/33782/Desktop/Animals10Captcha/Models/VGG_animals10.pkl"
GoogleNet_Path =  "C:/Users/33782/Desktop/Animals10Captcha/Models/GoogLeNet_animals10.pkl"
AlexNet_Path = "C:/Users/33782/Desktop/Animals10Captcha/Models/AlexNet_animals10.pkl"

###################################################################

class_dictionary = {'butterfly': 0, 'cat': 1, 'chicken':2 , 'cow': 3,
                    'dog': 4, 'elephant': 5, 'horse': 6, 'sheep': 7,
                    'spider': 8, 'squirrel': 9}
index_dictionary = { 0:'butterfly' , 1:'cat', 2:'chicken' , 3: 'cow',
                     4:'dog', 5:'elephant', 6:'horse', 7:'sheep',
                     8:'spider', 9:'squirrel'}

if model == "ResNet":
    model = pkl.load(open(ResNet_path, "rb"))
elif model == "VGG":
    model = pkl.load(open(VGG_path, "rb"))
elif model == "GoogleNet":
    model = pkl.load(open(GoogleNet_Path, "rb"))
elif model == "AlexNet":
    model = pkl.load(open(AlexNet_Path, "rb"))
else:
    raise Exception("No model selected")


acc = 0
for image in os.listdir(image_folder):

    img = Image.open(image_folder + image)
    img = np.asarray(img)
    img = img[:, :, :3]

    real_class_name = image.split("_")[0]
    real_class = class_dictionary[real_class_name]

    if filter:
        img = cv.medianBlur(img, 3)

    probabilities = model.predict_numpy_images([img])
    print("Image: ", image)
    print(probabilities)
    predicted_class = int(np.argmax(probabilities))
    predicted_class_name = index_dictionary[predicted_class]

    print(real_class, predicted_class)

    if real_class == predicted_class:
        acc = acc + 1

print(acc/len(os.listdir(image_folder)))



