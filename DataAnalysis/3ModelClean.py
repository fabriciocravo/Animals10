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
image_folder = "D:/Captcha_Data/ResNet_1000_Cleaned/"
save_folder = "D:/Captcha_Data/ResNet_1000_3ModelClean/"
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

resnet = pkl.load(open(ResNet_path, "rb"))
vgg = pkl.load(open(VGG_path, "rb"))
alexnet = pkl.load(open(AlexNet_Path, "rb"))

googlenet = pkl.load(open(GoogleNet_Path, "rb"))


acc = 0
for image in os.listdir(image_folder):

    img = Image.open(image_folder + image)
    img = np.asarray(img)
    img = img[:, :, :3]

    real_class_name = image.split("_")[0]
    real_class = class_dictionary[real_class_name]

    if filter:
        img = cv.medianBlur(img, 3)

    probabilities = resnet.predict_numpy_images([img])
    resnet_prediction = int(np.argmax(probabilities))

    probabilities = vgg.predict_numpy_images([img])
    vgg_prediction = int(np.argmax(probabilities))

    probabilities = alexnet.predict_numpy_images([img])
    alexnet_prediction = int(np.argmax(probabilities))

    if resnet_prediction != real_class and vgg_prediction != real_class and alexnet_prediction != real_class:
        plt.imsave(save_folder + image, img)

print(acc/len(os.listdir(image_folder)))



