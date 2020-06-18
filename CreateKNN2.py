import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def main():

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    pickle_in = open("X_encoded_30e_(4x40x30).pickle", "rb")
    X = pickle.load(pickle_in)
    X = X.reshape(len(X), 4800)

    pickle_in = open("y.pickle", "rb")
    y = pickle.load(pickle_in)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #X_train = np.array(X_train)
    #X_test = np.array(X_test)

    # instantiate learning model (k = 2)
    knn = KNeighborsClassifier(n_neighbors=2)

    # fitting the model
    knn.fit(X_train, y_train)

    # predict the response
    pred = knn.predict(X_test)

    # evaluate accuracy
    print("accuracy: {}".format(accuracy_score(y_test, pred)))


main()
