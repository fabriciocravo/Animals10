import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle as pkl

# Needed for pickle loading!!!
from ResNet import ResNet
from VGG import VGG
from GoogLeNet import GoogLeNet
from AlexNet import AlexNet

# Parameters ######################################################

model = "AlexNet"
image_folder = "D:/Captcha_Data/ResNet_1000_Acc_Test/"
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

mean_acc = 0
gaussian_acc = 0
median_acc = 0
billateral_acc = 0
for image in os.listdir(image_folder):

    img = Image.open(image_folder + image)
    img = np.asarray(img)
    img = img[:, :, :3]

    real_class = image.split("_")[0]
    real_class = class_dictionary[real_class]

    filter_img = cv2.blur(img, (3,3))
    predicted_class = int(np.argmax(model.predict_numpy_images([filter_img])))
    if real_class == predicted_class:
        mean_acc = mean_acc + 1

    filter_img = cv2.GaussianBlur(img, (3, 3), 0)
    predicted_class = int(np.argmax(model.predict_numpy_images([filter_img])))
    if real_class == predicted_class:
        gaussian_acc = gaussian_acc + 1

    filter_img = cv2.medianBlur(img, 3)
    predicted_class = int(np.argmax(model.predict_numpy_images([filter_img])))
    if real_class == predicted_class:
        median_acc = median_acc + 1

    filter_img = cv2.bilateralFilter(img, 9, 200, 200)
    predicted_class = int(np.argmax(model.predict_numpy_images([filter_img])))
    if real_class == predicted_class:
        billateral_acc = billateral_acc + 1

print("Mean: ", mean_acc/len(os.listdir(image_folder)))
print("Gaussian: ", gaussian_acc/len(os.listdir(image_folder)))
print("Median: ", median_acc/len(os.listdir(image_folder)))
print('Bilateral: ', billateral_acc/len(os.listdir(image_folder)))



