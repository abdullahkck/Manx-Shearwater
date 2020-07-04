import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from datetime import datetime


def main():
    num_folds = 10
    NAME = "Large-B006-vs-B007-" + str(num_folds) + "-folds-KNN"

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    #  dataset_name = "Calls_67_X_encoded_30e_(4x40x30)"
    dataset_name = "Calls_67_X_encoded_30e_(4x40x30)"

    pickle_in = open(dataset_name + ".pickle", "rb")
    X = pickle.load(pickle_in)
    X = X.reshape(len(X), 4800)

    #  pickle_in = open("Calls_67_y.pickle", "rb")
    pickle_in = open("Calls_67_y.pickle", "rb")
    y = pickle.load(pickle_in)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    inputs = np.concatenate((X_train, X_test), axis=0)
    targets = np.concatenate((y_train, y_test), axis=0)

    k_fold = KFold(n_splits=num_folds, shuffle=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.figure(1)
    plt.title(NAME + ' ROC curves')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    print(".....Training started.....")
    start_time = datetime.now()

    best_auc_score_per_fold = []
    best_acc_score_per_fold = []
    best_loss_score_per_fold = []

    for train, test in k_fold.split(inputs, targets):

        K = []
        training_scores = []
        test_scores = []
        auc_scores = []
        fpr_scores = []
        tpr_scores = []
        scores = {}

        for k in range(1, 10):
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(inputs[train], targets[train])

            # get AUC score
            y_scores = clf.predict_proba(inputs[test])
            fpr, tpr, threshold = roc_curve(targets[test], y_scores[:, 1])
            auc_score = auc(fpr, tpr)

            tpr_scores.append(tpr)
            fpr_scores.append(fpr)
            auc_scores.append(auc_score)

            # get training and test scores
            training_score = clf.score(inputs[train], targets[train])
            test_score = clf.score(inputs[test], targets[test])

            training_scores.append(training_score)
            test_scores.append(test_score)

            K.append(k)
            scores[k] = [training_score, test_score, '%0.3f' % auc_score]

        print('----------------------------------')
        for keys, values in scores.items():
            print(keys, ':', values)

        # fig, ax = plt.subplots()
        # plt.title(dataset_name)
        # plt.xlabel('Cluster')
        # plt.ylabel('Score')
        # ax.scatter(K, training_scores, color='r', label='Training')
        # ax.scatter(K, test_scores, color='g', label='Test')
        # ax.scatter(K, auc_scores, color='y', label='AUC')
        # ax.legend()
        # plt.show()

        # get k with max AUC value
        max_k_index = auc_scores.index(max(auc_scores))
        best_auc_score_per_fold.append(auc_scores[max_k_index])
        best_acc_score_per_fold.append(test_scores[max_k_index])
        best_loss_score_per_fold.append(training_scores[max_k_index])

        plt.plot(fpr_scores[max_k_index], tpr_scores[max_k_index])

    print('------------------------------------------------------------------------')
    time_dif = datetime.now() - start_time
    print(".....Training finished.....")
    print("Folds: ", num_folds)
    print(NAME)
    print("Training time: ", time_dif)
    print('AUC scores:')
    for i in range(num_folds):
        print('Fold ' + str(i) + ': ', '%0.3f' % best_auc_score_per_fold[i])
    print('Average scores for all folds:')
    print('Accuracy:', '%0.3f' % np.mean(best_acc_score_per_fold))
    print('Std:', '%0.3f' % np.std(best_acc_score_per_fold))
    print('Loss:', '%0.3f' % np.mean(best_loss_score_per_fold))
    print('AUC:', '%0.3f' % np.mean(best_auc_score_per_fold))
    print('------------------------------------------------------------------------')

    ax.text(0.5, 0.01, 'Mean of AUCs = ' + '%0.3f' % np.mean(best_auc_score_per_fold), style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.show()
    fig.savefig(NAME + ".png")


main()



