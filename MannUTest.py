import scipy.stats as sts
import numpy as np


knn_auc_scores = [0.927, 0.940, 0.925, 0.925, 0.973, 0.957, 0.922, 0.941, 0.918, 0.921]
cnn_auc_scores = [0.901, 0.890, 0.877, 0.887, 0.880, 0.872, 0.872, 0.876, 0.905, 0.871]

result = sts.mannwhitneyu(knn_auc_scores, cnn_auc_scores, alternative='two-sided')
print('pvalue =', result[1])

