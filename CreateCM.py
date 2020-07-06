import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


array = [[28,  2,  3],
    [1, 15,  0],
    [0, 0,  3]]

df_cm = pd.DataFrame(array, index=[i for i in "123"], columns=[i for i in "123"])
fig, ax = plt.subplots()
#plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xlabel("Target")
ax.set_ylabel("Output")
plt.title("CNN Confusion Matrix of 3 Pairs")
plt.show()



