import numpy as np
import pylab as pl
import scipy.stats as st
from sklearn import linear_model

def corr_plot(x, y, max_num=10000, outlier=0.05, line_on=True,
    legend_on=True, size=10):
    score = st.pearsonr(x, y)[0]
    np.random.seed(0)
    if len(x) > max_num:
        idx = np.random.permutation(len(x))[:max_num]
        x, y = x[idx], y[idx]
    outlier = int(len(x) * outlier)
    
    xy = np.vstack([x,y])
    z = st.gaussian_kde(xy)(xy)
    idx = z.argsort()
    idx1, idx2 = idx[outlier:], idx[:outlier]
    
    pl.set_cmap('Blues')
    pl.scatter(x[idx1], y[idx1], c=np.log(z[idx1]*9+1), edgecolor='', s=size)
    pl.scatter(x[idx2], y[idx2], c="r", edgecolor='', s=size, alpha=0.7)
    pl.grid(alpha=0.3)

    if line_on:
        clf = linear_model.LinearRegression()
        clf.fit(x.reshape(-1,1), y)
        xx = np.linspace(x.min(), x.max(), 1000).reshape(-1,1)
        yy = clf.predict(xx)
        pl.plot(xx, yy, "k--", label="R=%.3f" %score)

    if legend_on:
        pl.legend(loc="best", fancybox=True, ncol=1)



def ROC_plot(state, scores, threshold=0.001, legend_on=True, label="predict", 
             base_line=True):
    """
    Plot ROC curve and calculate the Area under the curve (AUC) from the
    with the prediction scores and true labels.
    The threshold is the step of the ROC cureve.
    """
    thresholds = np.arange(0, 1+2*threshold, threshold)
    fpr, tpr = np.zeros(thresholds.shape[0]), np.zeros(thresholds.shape[0])
    for i in range(thresholds.shape[0]):
        idx = np.where(scores>=thresholds[i])[0]
        fpr[i] = np.sum(state[idx] == 0)/np.sum(state == 0).astype('float')
        tpr[i] = np.sum(state[idx] == 1)/np.sum(state == 1).astype('float')
                
    auc = 0
    for i in range(thresholds.shape[0]-1):
        auc = auc + (fpr[i]-fpr[i+1]) * (tpr[i]+tpr[i+1]) / 2.0
        
    pl.plot(fpr, tpr, "-", linewidth=2.0, label="%s: AUC=%.3f" %(label,auc))
    if base_line: pl.plot(np.arange(0,2), np.arange(0,2), "k--", linewidth=1.0,
        label="random: AUC=0.500")
        
    if legend_on:
        pl.legend(loc="best", fancybox=True, ncol=1)
    
    pl.xlabel("False Positive Rate (1-Specificity)")
    pl.ylabel("True Positive Rate (Sensitivity)")
    return fpr, tpr, thresholds, auc



