import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle as pkl

from ResNet import ResNet

# Parameters ######################################################

image_folder = "D:/Captcha_Data/ResNet_1000_Acc_Test/"
ResNet_path = "C:/Users/33782/Desktop/Animals10Captcha/Models/ResNet_animals10.pkl"

###################################################################

class_dictionary = {'butterfly': 0, 'cat': 1, 'chicken':2 , 'cow': 3,
                    'dog': 4, 'elephant': 5, 'horse': 6, 'sheep': 7,
                    'spider': 8, 'squirrel': 9}
index_dictionary = { 0:'butterfly' , 1:'cat', 2:'chicken' , 3: 'cow',
                     4:'dog', 5:'elephant', 6:'horse', 7:'sheep',
                     8:'spider', 9:'squirrel'}

resnet = pkl.load(open(ResNet_path, "rb"))

number_of_instaces = {}
number_of_instaces['butterfly'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
number_of_instaces['cat'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
number_of_instaces['chicken'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
number_of_instaces['cow'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
number_of_instaces['dog'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
number_of_instaces['elephant'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
number_of_instaces['horse'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
number_of_instaces['sheep'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
number_of_instaces['spider'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
number_of_instaces['squirrel'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for image in os.listdir(image_folder):

    img = Image.open(image_folder + image)
    img = np.asarray(img)
    img = img[:, :, :3]

    real_class = image.split("_")[0]
    predicted_class = int(np.argmax(resnet.predict_numpy_images([img])))

    number_of_instaces[real_class][predicted_class] = number_of_instaces[real_class][predicted_class] + 1

for real_class in number_of_instaces:
    percentages = []
    for e in number_of_instaces[real_class]:
        percentages.append(e/sum(number_of_instaces[real_class]))
    number_of_instaces[real_class] = percentages

for real_class in number_of_instaces:
    print(real_class, number_of_instaces[real_class])

exit()
for j,key in enumerate(number_of_instaces):
    for i, number in enumerate(number_of_instaces[key]):
        plt.bar(x=i,height=number)
    plt.xlabel(index_dictionary[j])
    plt.show()

