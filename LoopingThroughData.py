import os
import random as rd
from PIL import Image
import torch

path = "./raw-img/"
epochs = 1000
batch_size = 100


class_dictionary = {'butterfly': 0, 'cat': 1, 'chicken':2 , 'cow': 3,
                    'dog':4, 'elephant':5, 'horse': 6, 'sheep':7,
                    'spider': 8, 'squirrel':9}

validation_data = {}
training_data = []
for animal in os.listdir(path):
    validation_data[animal] = []
    for i,image_path in enumerate(os.listdir(path + animal)):

        if i >= 100:
            validation_data[animal].append(image_path)
        else:
            training_data.append([image_path, class_dictionary[animal]])



