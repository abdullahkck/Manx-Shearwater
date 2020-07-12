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
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


num_folds = 10
fold_no = 1
epoch = 25
NAME = "Pairs-(685_754)-(686_755)-(686_308)-" + str(num_folds) + "-folds-" + str(epoch) + "-epochs"
num_classes = 3
acc_per_fold = []
loss_per_fold = []
macro_auc_per_fold = []
weighted_auc_per_fold = []

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pickle_in = open("Calls_67_pairs_X.pickle", "rb")
#  pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Calls_67_pairs_y.pickle", "rb")
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

#  print class distribution
print(".....Class distributions.....")
unique, counts = np.unique(targets, return_counts=True)
print(dict(zip(unique, counts)))

# convert the training labels to categorical vectors
targets = to_categorical(targets, num_classes=3)

k_fold = StratifiedShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=0)

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
    model.add(Dense(3))

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

    y_prob = model.predict_proba(inputs[test])
    y_pred = model.predict_classes(inputs[test])
    # print("Test Predictions:")
    # print(y_prob)
    # print(y_pred)

    rounded_targets = np.argmax(targets[test], axis=1)

    cnf_matrix = confusion_matrix(rounded_targets, y_pred, labels=[0, 1, 2])
    # np.set_printoptions(precision=2)
    print("Current Fold:")
    print(fold_no)
    print(cnf_matrix)

    # plot_confusion_matrix(cm=cnf_matrix,
    #                       normalize=True,
    #                       target_names=['685_754', '686_755', '686_308'],
    #                       title="Confusion Matrix " + NAME)

    # for i in num_classes:
    #   roc_curve(targets[test][:, i], y_prob[:, 0])
    y_pred = to_categorical(y_pred, num_classes=3)
    macro_roc_auc    = roc_auc_score(targets[test], y_pred, average="macro")
    weighted_roc_auc = roc_auc_score(targets[test], y_pred, average="weighted")

    macro_auc_per_fold.append(macro_roc_auc)
    weighted_auc_per_fold.append(weighted_roc_auc)

    print("ROC AUC scores:\n{:.3f} (macro),\n{:.3f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc, weighted_roc_auc))

    fold_no = fold_no + 1

print('------------------------------------------------------------------------')
time_dif = datetime.now() - start_time
print(".....Training finished.....")
print("Epochs: ", epoch)
print("Folds: ", num_folds)
print(NAME)
print("Training time: ", time_dif)
print('Macro AUC scores:')
for i in range(num_folds):
    print('Fold ' + str(i) + ': ', '%0.3f' % macro_auc_per_fold[i])
print('Weighted AUC scores:')
for i in range(num_folds):
    print('Fold ' + str(i) + ': ', '%0.3f' % weighted_auc_per_fold[i])
print('Average scores for all folds:')
print('Accuracy:', '%0.3f' % np.mean(acc_per_fold))
print('Std:', '%0.3f' % np.std(acc_per_fold))
print('Loss:', '%0.3f' % np.mean(loss_per_fold))
print('Macro AUC:', '%0.3f' % np.mean(macro_auc_per_fold))
print('Weighted AUC:', '%0.3f' % np.mean(weighted_auc_per_fold))
print('------------------------------------------------------------------------')


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# for i in range(num_folds):
#     plt.plot(fpr_per_fold[i], tpr_per_fold[i])
#
# ax.text(0.5, 0.01, 'Mean of AUCs = ' + '%0.3f' % np.mean(auc_per_fold), style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title(NAME + ' ROC curves')
# plt.legend(loc='best')
# plt.show()
# fig.savefig(NAME + ".png")

