import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

path = '/Users/abdullahkucuk/Desktop/MSc/MSc Project/records.nosync/manx_shearwater_test_calls/Spectrograms'
training_data = []
categories = ["B006", "B007"]


def check_call_is_existed(img_array):
    if np.sum(img_array[0:480, 0:640]) > 13500000:  # if the sum of pixels less than 15 millions, there should be a call
        return True
    return False


def create_training_data():
    eliminated_spec_count = 0
    for img in tqdm(os.listdir(path)):  # iterate over each image
        try:
            category = img[:4]  # get the category name
            #  if category == "B006":
            #      continue
            class_num = categories.index(category)  # get the classification
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            if check_call_is_existed(img_array):
                training_data.append([img_array, class_num])  # add this to our training_data
            else:
                eliminated_spec_count = eliminated_spec_count + 1
                # print("Eliminated img name: ", img, " sum: ", np.sum(img_array[0:480, 0:640]))
        except Exception as e:
            print("Image read exception: ", e, os.path.join(path, img))
            pass
    print("Training data count: ", len(training_data))
    print("Eliminated data count: ", eliminated_spec_count)


def create_pickles():
    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, 640, 480, 1)

    #  pickle_out = open("X_B007.pickle", "wb")
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    #  pickle_out = open("y_B007.pickle", "wb")
    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def main():

    create_training_data()
    random.shuffle(training_data)
    create_pickles()


main()

