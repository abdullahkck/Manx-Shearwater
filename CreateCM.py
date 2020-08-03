import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


normalize = False
title = "3 Pairs CNN Classification Trained with Original Samples"

sum = [[0,  0,  0],
    [0, 0,  0],
    [0,  0,  0]]


fold1 = [[289,   3,   1],
 [ 12, 159,   2],
 [ 23,   1,  32]]

fold2 = [[283,   8,   2],
 [  8, 162,   3],
 [ 20,   9,  27]]

fold3 = [[282,   6,   5],
 [ 18, 151,   4],
 [ 29,   5,  22]]

fold4 = [[273,  12,   8],
 [  3, 166,   4],
 [ 14,   4,  38]]

fold5 = [[284,   5,   4],
 [ 11, 159,   3],
 [ 14,   7,  35]]



all_folds = [fold1, fold2, fold3, fold4, fold5]

for fold in all_folds:
    sum = np.add(sum, fold)

# sum = [[505,  32,  46],
#  [ 23, 303,  21],
#  [  7,  20, 333]]





cm = pd.DataFrame(sum,
                     index=[i for i in ["685_754", "686_755", "686_308"]],
                     columns=[i for i in ["685_754", "686_755", "686_308"]],
                  )

fmt = ".0f"
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    title = "Normalized " + title
    fmt = '.2f'
fig, ax = plt.subplots()
#plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt=fmt)

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.title(title)
plt.show()



