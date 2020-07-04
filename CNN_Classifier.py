import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import pickle
from keras.callbacks import TensorBoard
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split


num_folds = 10
fold_no = 1
epoch = 25
NAME = "Large-B006-vs-B007-" + str(num_folds) + "-folds-" + str(epoch) + "-epochs"
num_classes = 2
acc_per_fold = []
loss_per_fold = []
auc_per_fold = []
fpr_per_fold = []
tpr_per_fold = []

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pickle_in = open("Calls_67_X.pickle", "rb")
#  pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Calls_67_y.pickle", "rb")
#  pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = X_train/255.0
X_test = X_test/255.0

inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

k_fold = KFold(n_splits=num_folds, shuffle=True)

print(".....Training started.....")
start_time = datetime.now()

for train, test in k_fold.split(inputs, targets):
    print("Current Fold: ", fold_no)
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

    # model.summary()

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(inputs[train], targets[train], batch_size=128, epochs=epoch, validation_split=0.2, callbacks=[tensorboard])

    # Test vs Training
    test_eval = model.evaluate(inputs[test], targets[test], verbose=0)
    acc_per_fold.append(test_eval[1])
    loss_per_fold.append(test_eval[0])

    print('Test accuracy:', test_eval[1])
    print('Test loss:', test_eval[0])

    # AUC
    y_pred = model.predict(inputs[test]).ravel()
    fpr, tpr, thresholds = roc_curve(targets[test], y_pred)
    _auc = auc(fpr, tpr)
    auc_per_fold.append(_auc)
    fpr_per_fold.append(fpr)
    tpr_per_fold.append(tpr)

    print('AUC:', '%0.3f' % _auc)

    print("Test Predictions:")
    print(model.predict_proba(inputs[test]))

    fold_no = fold_no + 1

print('------------------------------------------------------------------------')
time_dif = datetime.now() - start_time
print(".....Training finished.....")
print("Epochs: ", epoch)
print("Folds: ", num_folds)
print(NAME)
print("Training time: ", time_dif)
print('AUC scores:')
for i in range(num_folds):
    print('Fold ' + str(i) + ': ', '%0.3f' % auc_per_fold[i])
print('Average scores for all folds:')
print('Accuracy:', '%0.3f' % np.mean(acc_per_fold))
print('Std:', '%0.3f' % np.std(acc_per_fold))
print('Loss:', '%0.3f' % np.mean(loss_per_fold))
print('AUC:', '%0.3f' % np.mean(auc_per_fold))
print('------------------------------------------------------------------------')


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
for i in range(num_folds):
    plt.plot(fpr_per_fold[i], tpr_per_fold[i])

ax.text(0.5, 0.01, 'Mean of AUCs = ' + '%0.3f' % np.mean(auc_per_fold), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(NAME + ' ROC curves')
plt.legend(loc='best')
plt.show()
fig.savefig(NAME + ".png")

