import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


normalize = True
title = "CM of 3 Pairs CNN Classification Sum"

sum = [[0,  0,  0],
    [0, 0,  0],
    [0,  0,  0]]

fold0 = [[57,  1,  1],
 [ 1, 32 , 2],
 [ 6 , 0 , 5]]

fold1 = [[54,  3 , 2],
 [ 2, 32,  1],
 [ 4,  1 , 6]]

fold2 = [[58, 1,  0],
 [ 3, 32,  0],
 [ 5,  4,  2]]

fold3 = [[48,  3,  8],
 [ 1, 31,  3],
 [ 2 , 0 , 9]]

fold4 = [[54,  1 , 4],
 [ 2, 32,  1],
 [ 4 , 3,  4]]

fold5 = [[56  ,2,  1],
 [ 2, 33 , 0],
 [ 6 , 3,  2]]

fold6 = [[43 , 8 , 8],
 [ 0, 35  ,0],
 [ 0 , 4,  7]]

fold7 = [[56,  1,  2],
 [ 1, 33,  1],
 [ 7 , 2 , 2]]

fold8 = [[56,  2,  1],
 [ 2 ,33  ,0],
 [ 3  ,3 , 5]]

fold9 = [[51,  6,  2],
 [ 1, 34,  0],
 [ 4,  3,  4]]

all_folds = [fold0, fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9]

for fold in all_folds:
    sum = np.add(sum, fold)

# sum = [[58,  1,  0],
#  [ 2, 33,  0],
#  [ 1,  3,  7]]

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



