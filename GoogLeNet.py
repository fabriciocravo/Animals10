# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random as rd
import pickle as pkl

# for reading and displaying images
from PIL import Image
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import Adam, SGD
import torch.optim as optim
from torch.utils.data import DataLoader

# torchvision for pre-trained models
from torchvision import models, transforms


class GoogLeNet:

    """
            Define transforms for the data
    """
    preprocess_training_data = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    preprocess_data = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    preprocess_numpy_data = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self):
        self.model = models.googlenet(pretrained=True)

        # Freeze the parameters
        for _, param in enumerate(self.model.parameters()):
            param.requires_grad = False
            if _ == 200:
                break

        self.model.fc = Sequential(
            Linear(1024, 10),
            Softmax(dim=1))

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            raise Exception("Gotta make that GPU work!!!!!")

        self.criterion = NLLLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)


        self.device = torch.device("cuda:0")

    def train_net(self, train_x, train_y):

        self.model.train()

        train_x, train_y = train_x.to(device=self.device), train_y.to(device=self.device)

        self.optimizer.zero_grad()

        output = self.model(train_x)
        loss = self.criterion(output, train_y)
        loss.backward()
        self.optimizer.step()

        print(loss)


    def train_single(self, train_x, train_y):
        pass

    def predict_single(self, img):

        self.model.eval()

        img = self.preprocess_data(img)
        img = img.view(-1, 3, 100, 100)
        img = img.to(self.device)

        probabilities = self.model(img)
        probabilities = probabilities.cpu().detach().numpy()
        print(probabilities)
        predicted_class = np.argmax(probabilities)

        return probabilities, predicted_class


    def predict_numpy_images(self, imgs):

        self.model.eval()

        processed_imgs = []
        for img in imgs:
            img = self.preprocess_numpy_data(img)
            processed_imgs.append(img.unsqueeze(0))
        processed_imgs = torch.cat(processed_imgs)
        processed_imgs = processed_imgs.to(self.device)

        probabilities = self.model(processed_imgs)

        return probabilities.cpu().detach().numpy()






if __name__ == '__main__':

    googlenet = GoogLeNet()

    path = "./raw-img/"
    epochs = 10000
    batch_size = 100

    class_dictionary = {'butterfly': 0, 'cat': 1, 'chicken': 2, 'cow': 3,
                        'dog': 4, 'elephant': 5, 'horse': 6, 'sheep': 7,
                        'spider': 8, 'squirrel': 9}

    validation_data = {}
    training_data = []
    for animal in os.listdir(path):
        validation_data[animal] = []
        for i, image_path in enumerate(os.listdir(path + animal)):

            path_img = path + animal + "/" + image_path
            if i < 100:
                validation_data[animal].append(path_img)
            else:
                training_data.append([path_img, class_dictionary[animal]])

    for _ in range(epochs):
        train_x = []
        train_y = []

        for i in range(batch_size):

            try:
                path, y = rd.choice(training_data)
                img = Image.open(path)
                img = googlenet.preprocess_training_data(img)
                train_x.append(img.unsqueeze(0))
                train_y.append(y)
            except:
                pass

        train_x = torch.cat(train_x)
        train_y = torch.LongTensor(train_y)

        googlenet.train_net(train_x, train_y)

    total = 0
    acc = 0
    for i, key in enumerate(validation_data):
        list_of_image_paths = validation_data[key]
        for image_path in list_of_image_paths:

            try:
                img = Image.open(image_path)
                total = total + 1
                _, predicted_class = googlenet.predict_single(img)
                print(predicted_class)

                if predicted_class == i:
                    acc = acc + 1
            except:
                pass

    print("The final accuracy is: ")
    print(acc / total)

    pkl.dump(googlenet, open("./Models/GoogLeNet_animals10.pkl", "wb"))

