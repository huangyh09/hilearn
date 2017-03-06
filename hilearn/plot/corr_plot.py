import numpy as np
import pylab as pl
import scipy.stats as st
from sklearn import linear_model

def corr_plot(x, y, max_num=10000, outlier=0.05, line_on=True,
              corr_on=True, size=30, dot_color=None, outlier_color="r",
              alpha=0.8): #"deepskyblue"
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
        c_score = np.log2(z[idx]+100)
    else:
        #idx2 = []
        c_score = dot_color
    
    pl.set_cmap("Blues")
    pl.scatter(x[idx], y[idx], c=c_score, edgecolor='', s=size, alpha=alpha)
    pl.scatter(x[idx2], y[idx2], c=outlier_color, edgecolor='', s=size/5, alpha=alpha/3.0)#/5
    pl.grid(alpha=0.3)

    if line_on:
        clf = linear_model.LinearRegression()
        clf.fit(x.reshape(-1,1), y)
        xx = np.linspace(x.min(), x.max(), 1000).reshape(-1,1)
        yy = clf.predict(xx)
        pl.plot(xx, yy, "k--", label="R=%.3f" %score[0])
        # pl.plot(xx, yy, "k--")

    if corr_on:
        pl.legend(loc="best", fancybox=True, ncol=1)
        # pl.annotate("R=%.3f\np=%.1e" %score, fontsize='x-large', 
        #             xy=(0.97, 0.05), xycoords='axes fraction',
        #             textcoords='offset points', ha='right', va='bottom')



# def ROC_plot(state, scores, threshold=0.001, legend_on=True, label="predict", 
#              base_line=True):
#     """
#     Plot ROC curve and calculate the Area under the curve (AUC) from the
#     with the prediction scores and true labels.
#     The threshold is the step of the ROC cureve.
#     """
#     thresholds = np.arange(0, 1+2*threshold, threshold)
#     fpr, tpr = np.zeros(thresholds.shape[0]), np.zeros(thresholds.shape[0])
#     for i in range(thresholds.shape[0]):
#         idx = np.where(scores>=thresholds[i])[0]
#         fpr[i] = np.sum(state[idx] == 0)/np.sum(state == 0).astype('float')
#         tpr[i] = np.sum(state[idx] == 1)/np.sum(state == 1).astype('float')
                
#     auc = 0
#     for i in range(thresholds.shape[0]-1):
#         auc = auc + (fpr[i]-fpr[i+1]) * (tpr[i]+tpr[i+1]) / 2.0
        
#     pl.plot(fpr, tpr, "-", linewidth=2.0, label="%s: AUC=%.3f" %(label,auc))
#     if base_line: pl.plot(np.arange(0,2), np.arange(0,2), "k--", linewidth=1.0,
#         label="random: AUC=0.500")
        
#     if legend_on:
#         pl.legend(loc="best", fancybox=True, ncol=1)
    
#     pl.xlabel("False Positive Rate (1-Specificity)")
#     pl.ylabel("True Positive Rate (Sensitivity)")
#     return fpr, tpr, thresholds, auc


def ROC_plot(state, scores, threshold=None, color=None, legend_on=True, label="predict", 
             base_line=True, linewidth=1.0):
    """
    Plot ROC curve and calculate the Area under the curve (AUC) from the
    with the prediction scores and true labels.
    The threshold is the step of the ROC cureve.
    """
    # if color is None or color=="none": 
    #     color = np.random.rand(3,1)
    
    score_gap = np.unique(scores)
    if len(score_gap) > 2000:
        idx = np.random.permutation(len(score_gap))
        score_gap = score_gap[idx[:2000]]
    score_gap = np.append(np.min(score_gap)-0.1, score_gap)
    score_gap = np.append(score_gap, np.max(score_gap)+0.1)
    if threshold is not None:
        thresholds = np.sort(np.append(threshold, score_gap))
        _idx = np.where(scores >= threshold)[0]
        _fpr = np.sum(state[_idx] == 0)/np.sum(state == 0).astype('float')
        _tpr = np.sum(state[_idx] == 1)/np.sum(state == 1).astype('float')
        # pl.scatter(_fpr, _tpr, marker="o", s=80, facecolors='none', edgecolors=color)
        if color is None:
            pl.plot(_fpr, _tpr, marker='o', markersize=8, mfc='none')
        else:
            pl.plot(_fpr, _tpr, marker='o', markersize=8, mec=color, mfc=color) 
    else:
        thresholds = np.sort(score_gap)
    #thresholds = np.arange(np.min(threshold), 1+2*threshold, threshold)
    
    fpr, tpr = np.zeros(thresholds.shape[0]), np.zeros(thresholds.shape[0])
    for i in range(thresholds.shape[0]):
        idx = np.where(scores >= thresholds[i])[0]
        fpr[i] = np.sum(state[idx] == 0)/np.sum(state == 0).astype('float')
        tpr[i] = np.sum(state[idx] == 1)/np.sum(state == 1).astype('float')
        
    auc = 0
    for i in range(thresholds.shape[0]-1):
        auc = auc + (fpr[i]-fpr[i+1]) * (tpr[i]+tpr[i+1]) / 2.0
        
    if color is None:
        pl.plot(fpr, tpr, "-",  linewidth=linewidth,
            label="%s: AUC=%.3f" %(label,auc))
    else:
        pl.plot(fpr, tpr, "-",  linewidth=linewidth, color=color,
            label="%s: AUC=%.3f" %(label,auc))
    if base_line: pl.plot(np.arange(0,2), np.arange(0,2), "k--", linewidth=1.0,
        label="random: AUC=0.500")
        
    if legend_on:
        pl.legend(loc="best", fancybox=True, ncol=1)
    
    pl.xlabel("False Positive Rate (1-Specificity)")
    pl.ylabel("True Positive Rate (Sensitivity)")
    return fpr, tpr, thresholds, auc


def PR_curve(state, scores, threshold=None, color=None, legend_on=True,  
             label="predict", base_line=False, linewidth=1.5):
    """
    Plot ROC curve and calculate the Area under the curve (AUC) from the
    with the prediction scores and true labels.
    The threshold is the step of the ROC cureve.
    """
    if color is None or color=="none": 
        color = np.random.rand(3,1)
    
    score_gap = np.unique(scores)
    if len(score_gap) > 2000:
        idx = np.random.permutation(len(score_gap))
        score_gap = score_gap[idx[:2000]]
    #score_gap = np.append(np.min(score_gap)-0.1, score_gap)
    #score_gap = np.append(score_gap, np.max(score_gap)+0.1)
    if threshold is not None:
        thresholds = np.sort(np.append(threshold, score_gap))
        idx1 = np.where(scores >= threshold)[0]
        idx2 = np.where(scores <  threshold)[0]
        FP = np.sum(state[idx1] == 0)
        TP = np.sum(state[idx1] == 1)
        FN = np.sum(state[idx2] == 1)
        _pre = (TP+0.0)/(TP + FP)
        _rec = (TP+0.0)/(TP + FN)
        # pl.plot(_rec, _pre, marker='*', markersize=9, mec="k", mfc='none')
        pl.plot(_rec, _pre, marker='o', markersize=8, mec=color, mfc=color)
    else:
        thresholds = np.sort(score_gap)
    
    pre, rec = np.zeros(thresholds.shape[0]), np.zeros(thresholds.shape[0])
    for i in range(thresholds.shape[0]):
        idx1 = np.where(scores >= thresholds[i])[0]
        idx2 = np.where(scores <  thresholds[i])[0]
        FP = np.sum(state[idx1] == 0)
        TP = np.sum(state[idx1] == 1)
        FN = np.sum(state[idx2] == 1)
        pre[i] = (TP+0.0)/(TP + FP)
        rec[i] = (TP+0.0)/(TP + FN)
        
    auc = 0
    _rec = np.append(1.0, rec)
    _pre = np.append(0.0, pre)
    _rec = np.append(_rec, 0.0)
    _pre = np.append(_pre, 1.0)
    for i in range(_rec.shape[0]-1):
        auc = auc + (_rec[i]-_rec[i+1]) * (_pre[i]+_pre[i+1]) / 2.0
        
    pl.plot(_rec, _pre, "-", color=color, linewidth=linewidth, label="%s: AUC=%.3f" %(label,auc))
    if base_line: pl.plot(np.arange(0,2), 1-np.arange(0,2), "k--", linewidth=1.0,
        label="random: AUC=0.500")
        
    if legend_on:
        pl.legend(loc="best", fancybox=True, ncol=1)
    
    pl.ylabel("Precision: TP/(TP+FP)")
    pl.xlabel("Recall: TP/(TP+FN)")
    return rec, pre, thresholds, auc
