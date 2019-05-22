import numpy as np
import matplotlib.pyplot as plt

def ROC_plot(state, scores, threshold=None, color=None, legend_on=True, 
             label="predict", base_line=True, linewidth=1.0):
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
        # plt.scatter(_fpr, _tpr, marker="o", s=80, facecolors='none', edgecolors=color)
        if color is None:
            plt.plot(_fpr, _tpr, marker='o', markersize=8, mfc='none')
        else:
            plt.plot(_fpr, _tpr, marker='o', markersize=8, mec=color, mfc=color) 
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
        plt.plot(fpr, tpr, "-",  linewidth=linewidth,
                 label="%s: AUC=%.3f" %(label,auc))
    else:
        plt.plot(fpr, tpr, "-",  linewidth=linewidth, color=color,
                 label="%s: AUC=%.3f" %(label,auc))
    if base_line: plt.plot(np.arange(0,2), np.arange(0,2), "k--", linewidth=1.0,
        label="random: AUC=0.500")
        
    if legend_on:
        plt.legend(loc="best", fancybox=True, ncol=1)
    
    plt.xlabel("False Positive Rate (1-Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    return fpr, tpr, thresholds, auc


def PR_curve(state, scores, threshold=None, color=None, legend_on=True,  
             label="predict", base_line=False, linewidth=1.5):
    """
    Plot ROC curve and calculate the Area under the curve (AUC) from the
    with the prediction scores and true labels.
    The threshold is the step of the ROC cureve.
    """

    ###Test compare
    # from sklearn.metrics import precision_recall_curve,average_precision_score
    # precision, recall, thresholds = precision_recall_curve(labels, BF_tmp)
    # ap = average_precision_score(labels, BF_tmp)
    # plt.plot(recall, precision, label="%.3f" %(ap))

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
        # plt.plot(_rec, _pre, marker='*', markersize=9, mec="k", mfc='none')
        plt.plot(_rec, _pre, marker='o', markersize=8, mec=color, mfc=color)
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
        
    plt.plot(_rec, _pre, "-", color=color, linewidth=linewidth, 
             label="%s: AUC=%.3f" %(label,auc))
    if base_line: plt.plot(np.arange(0,2), 1-np.arange(0,2), "k--", 
                           linewidth=1.0, label="random: AUC=0.500")
        
    if legend_on:
        plt.legend(loc="best", fancybox=True, ncol=1)
    
    plt.ylabel("Precision: TP/(TP+FP)")
    plt.xlabel("Recall: TP/(TP+FN)")
    return rec, pre, thresholds, auc



def ecdf_plot(data, x=None, **kwargs):
    """
    Empirical plot for cumulative distribution function
    
    Parameters
    ----------
    data: array or list
        data for the empirical CDF plot
    x: array or list (optional)
        the points to show the plot
    **kwargs: 
        **kwargs for matplotlib.plot
        
    Returns
    -------
    x: array
        sorted x
    ecdf_val:
        values of empirical cdf for x
    """
    data = np.sort(np.array(data))
    if x is None:
        x = data
    else:
        x = np.sort(np.array(x))
        
    ecdf_val = np.zeros(len(x))
    for i in range(len(x)):
        ecdf_val[i] = np.mean(data < x[i])
    
    plt.plot(x, ecdf_val, **kwargs)
    return x, ecdf_val