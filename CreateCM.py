import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


normalize = True
title = "5 Burrows kNN Classification 10-folds"

sum = [[ 44 ,  2,   2,   9,  13],
 [  0 , 77,   2,   1,   0],
 [  2 ,  3, 199,   5,  11],
 [  7 ,  1,  21, 474,  27],
 [ 10 ,  0,  10,  19, 351]]

cm = pd.DataFrame(sum,
                     index=[i for i in ["B001", "B003", "B004", "B006", "B007"]],
                     columns=[i for i in ["B001", "B003", "B004", "B006", "B007"]],
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



