import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

path = '/Users/abdullahkucuk/Desktop/MSc/MSc Project/records.nosync/calls_all/Spectrograms'
training_data = []
categories = ["B001", "B003", "B004", "B006", "B007"]


def check_call_is_existed(img_array):
    if np.sum(img_array[0:480, 0:640]) > 13500000:  # if the sum of pixels less than 15 millions, there should be a call
        return True
    return False


def create_training_data():
    eliminated_samples = [0, 0, 0, 0, 0]
    training_samples = [0, 0, 0, 0, 0]

    for img in tqdm(os.listdir(path)):  # iterate over each image
        try:
            category = img[:4]  # get the category name
            # if category == "B007":
            #     continue
            class_num = categories.index(category)  # get the classification
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            if check_call_is_existed(img_array):
                training_samples[class_num] = training_samples[class_num] + 1
                training_data.append([img_array, class_num])  # add this to our training_data
            else:
                eliminated_samples[class_num] = eliminated_samples[class_num] + 1
                eliminated_spec_count = eliminated_spec_count + 1
                # print("Eliminated img name: ", img, " sum: ", np.sum(img_array[0:480, 0:640]))
        except Exception as e:
            print("Image read exception: ", e, os.path.join(path, img))
            pass
    print("Total training samples count: ", len(training_data))
    print("Training sample counts: ")
    print(training_samples)
    print("Eliminated sample counts: ")
    print(eliminated_samples)


def create_pickles():
    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, 640, 480, 1)

    #  pickle_out = open("Calls_67_X_B006.pickle", "wb")
    #  pickle_out = open("Calls_67_X.pickle", "wb")
    pickle_out = open("Calls_all_X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    #  pickle_out = open("Calls_67_y_B006.pickle", "wb")
    #  pickle_out = open("Calls_67_y.pickle", "wb")
    pickle_out = open("Calls_all_y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def main():

    create_training_data()
    random.shuffle(training_data)
    create_pickles()


main()

