import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


normalize = True
title = "CM of Balanced 3 Pairs kNN Classification Sum"

sum = [[0,  0,  0],
    [0, 0,  0],
    [0,  0,  0]]

fold1 = [[54,  0,  4],
 [ 2, 32,  1],
 [ 9,  2, 25]]

fold2 = [[47,  5,  6],
 [ 1, 33,  1],
 [ 0,  1, 35]]

fold3 = [[51,  6,  2],
 [ 0, 34,  0],
 [ 4,  5, 27]]

fold4 = [[54,  3,  2],
 [ 7, 27,  0],
 [ 5,  2, 29]]

fold5 = [[52,  3,  3],
 [ 2, 32,  1],
 [ 5,  1, 30]]

fold6 = [[49,  3,  6],
 [ 0, 32,  3],
 [ 2,  0, 34]]

fold7 = [[49,  5,  4],
 [ 1, 34,  0],
 [ 5,  5, 26]]

fold8 = [[50,  1,  7],
 [ 1, 28,  6],
 [ 5,  1, 30]]

fold9 = [[56,  0,  3],
 [ 1, 32,  1],
 [ 3,  3, 30]]

fold10 = [[47,  1, 10],
 [ 0, 31,  4],
 [ 1,  2, 33]]

all_folds = [fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10]

for fold in all_folds:
    sum = np.add(sum, fold)

sum = [[505,  32,  46],
 [ 23, 303,  21],
 [  7,  20, 333]]



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



