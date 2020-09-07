import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt


# plt_title = 'kNN 10 folds 3 Pairs Classification Original vs Balanced Dataset'
plt_title = '3 Pairs classification CNN vs kNN best results'
#labels = ('Original', 'Balanced')
labels = ('CNN balanced', 'kNN original')
first_auc_scores = [
0.885,
0.923,
0.903,
0.877,
0.908,
0.922,
0.883,
0.877,
0.934,
0.902
]

second_auc_scores = [
0.927,
0.940,
0.925,
0.925,
0.973,
0.957,
0.922,
0.941,
0.918,
0.921
]

result = sts.mannwhitneyu(second_auc_scores, first_auc_scores, alternative='two-sided')
p_value = result[1]
print('pvalue =', result[1])

scoreMeans   = (np.mean(first_auc_scores), np.mean(second_auc_scores))
scoreStd     = (np.std(first_auc_scores), np.std(second_auc_scores))
ind  = [1, 2]    # the x locations for the groups
width= 0.6

# Pull the formatting out here
bar_kwargs = {'width':width,'color':'b','linewidth':2,'zorder':5}
err_kwargs = {'zorder':0,'fmt':'none','linewidth':2,'ecolor':'k'}

fig, ax = plt.subplots()
ax.p1 = plt.bar(ind, scoreMeans, **bar_kwargs)
ax.errs = plt.errorbar(ind, scoreMeans, yerr=scoreStd, **err_kwargs)


# Custom function to draw the diff bars

def label_diff(i,j,text,X,Y):
    x = (X[i]+X[j])/2
    y = 0.04*max(Y[i], Y[j])
    dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':1,'shrinkB':1.5,'linewidth':1}
    ax.annotate(text, xy=(X[i],y + 1.7), zorder=10)

    ax.annotate('std =' + str("%.3f" % np.std(first_auc_scores)), xy=(X[i], y + 0.92), zorder=10)
    ax.annotate('mean =' + str("%.3f" % np.mean(first_auc_scores)), xy=(X[i], y + 1.01), zorder=10)
    #ax.annotate('std =' + str("%.3f" % np.std(first_auc_scores)), xy=(X[i], y + 0.99), zorder=10)
    #ax.annotate('mean =' + str("%.3f" % np.mean(first_auc_scores)), xy=(X[i], y + 1.08), zorder=10)
    ax.annotate('std =' + str("%.3f" % np.std(second_auc_scores)), xy=(X[j], y + 0.99), zorder=10)
    ax.annotate('mean =' + str("%.3f" % np.mean(second_auc_scores)), xy=(X[j], y + 1.08), zorder=10)
    ax.annotate('', xy=(X[i], y + 1.2), xytext=(X[j], y + 1.2), arrowprops=props)

# Call the function
label_diff(0, 1,'p=' + str('%.5f' % p_value), ind, scoreMeans)

plt.title(plt_title)
plt.ylabel('Weighted AUC Mean')
plt.ylim(ymax=2)
plt.xticks(ind, labels, color='k')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1], color='k')
plt.show()
