# Import models
from ResNet import ResNet
# from VVG import VGG
# from GoogLeNet import GoogLeNet
# from AlexNet import AlexNet
import os

from scipy.optimize import differential_evolution
import numpy as np
from PIL import Image
import pickle as pkl
import matplotlib.pyplot as plt
import cv2 as cv
import random as rd

resnet = pkl.load(open("./Models/ResNet_animals10.pkl", "rb"))
# vgg = pkl.load(open("./Models/VGG_animals10.pkl", "rb"))
# alex = pkl.load(open("./Models/AlexNet_animals10.pkl", "rb"))
# goooglenet = pkl.load(open("./Models/GoogLeNet_cats_vs_dogs.pkl", "rb"))
class_dictionary = {'butterfly': 0, 'cat': 1, 'chicken':2 , 'cow': 3,
                    'dog': 4, 'elephant': 5, 'horse': 6, 'sheep': 7,
                    'spider': 8, 'squirrel': 9}


def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    imgs = []
    for i in range(len(xs)):
        imgs.append(np.copy(img))
    imgs = np.asarray(imgs)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        p = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):

                    if (img[i, j] + x[p] >= 255):
                        img[i, j] = 255
                    elif (img[i, j] + x[p] <= 0):
                        img[i, j] = 0
                    else:
                        img[i, j] = img[i, j] + x[p]
                    p = p + 1

    return imgs


def predict_classes(xs, img, target_class, model, minimize=True, filter=False):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)

    # Adding the filter to the prediction in order to make the attack more robust to filtering
    if filter:
        imgs_filtered = []
        for img in imgs_perturbed:
            imgs_filtered.append(cv.bilateralFilter(img, 9, 200, 200))
        imgs_perturbed = np.asarray(imgs_filtered)

    predictions = model.predict_numpy_images(imgs_perturbed)[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False, filter=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, img)[0]

    if filter:
        filter_attack_image = cv.bilateralFilter(attack_image, 9, 200, 200)
        filter_confidence = model.predict_numpy_images([filter_attack_image])[0]
        filter_prediction = np.argmax(filter_confidence)

    print("After the attack: ")
    print(attack_image.shape)
    confidence = model.predict_numpy_images([attack_image])[0]
    print(confidence)
    predicted_class = np.argmax(confidence)


    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    if not filter:
        if verbose:
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True
    else:
        if filter_prediction != target_class and predicted_class != target_class:
            return True


def attack(img, img_class, model, epsilon, target=None,
           maxiter=75, popsize=400, verbose=False, filter=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else class_dictionary[img_class]

    # Define bounds for every pixel value
    # Each pixel is accompanied by its epsilon value!!!!
    bounds = [(-epsilon, +epsilon)] * img.shape[0] * img.shape[1] * img.shape[2]


    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs, img, target_class,
                               model, target is None, filter)

    def callback_fn(x, convergence):
        return attack_success(x, img, target_class,
                              model, targeted_attack, verbose, filter)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, img)[0]

    """
        prior_probs = model.predict_one(x_test[img_id])
        predicted_probs = model.predict_one(attack_image)
        predicted_class = np.argmax(predicted_probs)
        actual_class = y_test[img_id, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]
    """

    # Show the best attempt at a solution (successful or not)
    #helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    #return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
            #predicted_probs, attack_result.x]
    return attack_image



path = "./raw-img/"
K = 200
for k in range(K):

    print(k/K*100)
    animal = rd.choice(os.listdir(path))
    image_name = rd.choice(os.listdir(path + animal))

    path_image = path + animal + "/" + image_name
    img = Image.open(path_image)
    img = img.resize((100, 100))
    img = np.asarray(img)

    attack_img = attack(img, animal, model=resnet, epsilon=10, maxiter=9000, popsize=10, filter=False)
    plt.imsave("./PerturbationResNet/{0}_{1}.png".format(animal, k), attack_img)