import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn import linear_model

def corr_plot(x, y, max_num=10000, outlier=0.01, line_on=True,
              corr_on=True, size=30, dot_color=None, outlier_color="r",
              alpha=0.8, color_rate=10): #"deepskyblue"
    score = st.pearsonr(x, y)
    np.random.seed(0)
    if len(x) > max_num:
        idx = np.random.permutation(len(x))[:max_num]
        x, y = x[idx], y[idx]
    outlier = int(len(x) * outlier)
    
    xy = np.vstack([x,y])
    z = st.gaussian_kde(xy)(xy)
    idx = z.argsort()
    idx1, idx2 = idx[outlier:], idx[:outlier]
    
    if dot_color is None: 
        #c_score = np.log2(z[idx]+100)
        c_score = np.log2(z[idx] + color_rate*np.min(z[idx]))
    else:
        #idx2 = []
        c_score = dot_color
    
    plt.set_cmap("Blues")
    plt.scatter(x[idx], y[idx], c=c_score, edgecolor='', s=size, alpha=alpha)
    plt.scatter(x[idx2], y[idx2], c=outlier_color, edgecolor='', s=size/5, 
                alpha=alpha/3.0)#/5
    plt.grid(alpha=0.4)

    if line_on:
        clf = linear_model.LinearRegression()
        clf.fit(x.reshape(-1,1), y)
        xx = np.linspace(x.min(), x.max(), 1000).reshape(-1,1)
        yy = clf.predict(xx)
        plt.plot(xx, yy, "k--", label="R=%.3f" %score[0])
        # plt.plot(xx, yy, "k--")

    if corr_on:
        plt.legend(loc="best", fancybox=True, ncol=1)
        # plt.annotate("R=%.3f\np=%.1e" %score, fontsize='x-large', 
        #             xy=(0.97, 0.05), xycoords='axes fraction',
        #             textcoords='offset points', ha='right', va='bottom')


def volcano_plot(fold_change, pval, p_min=0.00001, 
                 x_log10=True, p_threshold=0.05, 
                 h_color="red", label=None):
    """
    Volcano plot between log_fold change and p values, which is often used 
    for hypothesis test between two conditions.
    """
    pval[pval < p_min] = p_min
    idx1 = pval < p_threshold
    idx0 = pval >= p_threshold
    plt.scatter(fold_change[idx0], -np.log10(pval)[idx0], 
                color="grey", alpha=0.7, label=None)
    plt.scatter(fold_change[idx1], -np.log10(pval)[idx1], 
                color=h_color, alpha=0.7, label=label)
    plt.ylabel("-log10(p value)")
    plt.xlabel("Fold change")
    if x_log10: 
        plt.xscale('log', basex=10)