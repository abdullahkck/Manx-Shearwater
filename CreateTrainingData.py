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


def create_training_data():
    for img in tqdm(os.listdir(path)):  # iterate over each image
        try:
            category = img[:4]
            class_num = categories.index(category)  # get the classification
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            training_data.append([img_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass
        # except OSError as e:
        #    print("OSErrroBad img most likely", e, os.path.join(path,img))
        # except Exception as e:
        #    print("general exception", e, os.path.join(path,img))


create_training_data()
print(len(training_data))
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 640, 480, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
