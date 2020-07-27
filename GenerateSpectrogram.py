from keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# dataset_name = "Calls_67_pairs_X_encoded_30e_(4x40x30)"
#
# pickle_in = open(dataset_name + ".pickle", "rb")
# X = pickle.load(pickle_in)
# X = X.reshape(len(X), 4800)
#
# pickle_in = open("Calls_67_pairs_y.pickle", "rb")
# y = pickle.load(pickle_in)
#
# p3_x = []
# p3_y = []
#
# for idx, val in enumerate(y):
#     if val == 2:
#         p3_x = np.append(p3_x, X[idx])
#         p3_y = np.append(p3_y, val)
#
# p3_x = p3_x.reshape(len(p3_y), 4800)
#
# datagen = ImageDataGenerator(rotation_range=0,
#                                  shear_range=0.2,
#                                  zoom_range=0.2,
#                                  horizontal_flip=True)
#
# # fit parameters from data
# datagen.fit(p3_x)
#
# # datagen.flow(p3_x, p3_y, batch_size=9)
# # configure batch size and retrieve one batch of images
# for X_batch, y_batch in datagen.flow(p3_x, p3_y, batch_size=9):
#     # create a grid of 3x3 images
#     for i in range(0, 9):
#         plt.subplot(330 + 1 + i)
#         plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     # show the plot
#     plt.show()
#     break


def check_call_is_existed(img_array):
    if np.sum(img_array[0:480, 0:640]) > 13500000:  # if the sum of pixels more than 13.5 millions, there should be a call
        return True
    return False


path = '/Users/abdullahkucuk/Desktop/MSc/MSc Project/records.nosync/calls_67/Spectrograms_pairs'
training_data = []
save_path = '/Users/abdullahkucuk/Desktop/MSc/MSc Project/records.nosync/calls_67/686_308_generated'

eliminated_spec_count = 0
for img in tqdm(os.listdir(path)):  # iterate over each image
    try:
        category = img[:7]  # get the category name
        if category != "686_308":
            continue
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
        if check_call_is_existed(img_array):
            # load the image
            image = load_img(os.path.join(path, img))
            # convert to numpy array
            data = img_to_array(image)
            # expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(width_shift_range=0.5, horizontal_flip=True)
            # prepare iterator
            it = datagen.flow(samples, batch_size=1)
            # generate samples and plot
            for i in range(3):
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers for viewing
                gen_image = batch[0].astype('uint8')
                # plot raw pixel data
                fullpath = save_path + os.sep + img
                fullpath = fullpath[:-4] # remove .png
                fullpath = fullpath + '-' + str(i) +'.png'
                cv2.imwrite(fullpath, gen_image)

    except Exception as e:
        print("Image read exception: ", e, os.path.join(path, img))
        pass
print("Training data count: ", len(training_data))
print("Eliminated data count: ", eliminated_spec_count)