import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import pickle
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def printAccuracy():

    test_eval = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])


def drawROC():

    y_pred = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    _auc = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Roc curve (area = {:.3f})'.format(_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


NAME = "B006-vs-B007-CNN"
num_classes = 2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(640, 480, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 1)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 1)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(256))
model.add(Dense(32))
model.add(Dense(1))

model.add(Activation('sigmoid'))

#model.summary()

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(".....Training started.....")
start_time = datetime.now()
model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.2, callbacks=[tensorboard])
time_dif = datetime.now() - start_time
print(".....Training finished.....")
print("Training time: ", time_dif)

printAccuracy()
drawROC()



