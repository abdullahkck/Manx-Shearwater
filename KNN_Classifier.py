import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def main():

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    dataset_name = "X_encoded_30e_(4x40x30)"

    pickle_in = open(dataset_name + ".pickle", "rb")
    X = pickle.load(pickle_in)
    X = X.reshape(len(X), 4800)

    pickle_in = open("y.pickle", "rb")
    y = pickle.load(pickle_in)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #X_train = np.array(X_train)
    #X_test = np.array(X_test)

    K = []
    training_scores = []
    test_scores = []
    auc_scores = []
    fpr_scores = []
    tpr_scores = []
    scores = {}

    for k in range(1, 10):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        # get AUC score
        y_scores = clf.predict_proba(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
        auc_score = auc(fpr, tpr)

        tpr_scores.append(tpr)
        fpr_scores.append(fpr)
        auc_scores.append(auc_score)

        # get training and test scores
        training_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        training_scores.append(training_score)
        test_scores.append(test_score)

        K.append(k)
        scores[k] = [training_score, test_score, '%0.2f' % auc_score]

    for keys, values in scores.items():
        print(keys, ':', values)

    fig, ax = plt.subplots()
    plt.title(dataset_name)
    plt.xlabel('Cluster')
    plt.ylabel('Score')
    ax.scatter(K, training_scores, color='r', label='Training')
    ax.scatter(K, test_scores, color='g', label='Test')
    ax.scatter(K, auc_scores, color='y', label='AUC')
    ax.legend()
    plt.show()

    # get k with max AUC value
    max_k_index = auc_scores.index(max(auc_scores))

    # plot ROC curve for k with max AUC value
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_scores[max_k_index], tpr_scores[max_k_index], 'b', label='AUC = %0.2f' % auc_scores[max_k_index])
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve of kNN with ' + str(max_k_index + 1) + ' clusters')
    plt.show()


main()



