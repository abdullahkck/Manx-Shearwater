import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


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

    K = []
    training = []
    test = []
    scores = {}

    for k in range(1, 10):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        training_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        K.append(k)

        training.append(training_score)
        test.append(test_score)
        scores[k] = [training_score, test_score]

    for keys, values in scores.items():
        print(keys, ':', values)

    ax = sns.stripplot(K, training);
    ax.set(xlabel='values of k', ylabel='Training Score')

    plt.show()
    # function to show plot

    ax = sns.stripplot(K, test);
    ax.set(xlabel='values of k', ylabel='Test Score')
    plt.show()

    plt.title('kNN training vs testing scores')
    plt.xlabel('Cluster')
    plt.ylabel('Score')
    plt.scatter(K, training, color='k')
    plt.scatter(K, test, color='g')
    plt.show()
    # For overlapping scatter plots


main()
