import numpy as np
import matplotlib.pyplot as plt

def boxgroup(x, labels=None, conditions=None, colors=None, notch=False, sys='',
             widths=0.9, patch_artist=True, showmeans=True, alpha=1, **kwargs):
    """
    Make boxes for a multiple groups data in different conditions.
    
    Parameters
    ----------
    x: a list of multiple groups
        The input data, e.g., [group1, group2, ..., groupN]. 
        If there is only one group, use [group1]. For each gorup,
        it can be an array or a list, containing the same number of conditions.
    labels: a list or an array
        The names of each group memeber
    conditions: a list or an array
        The names of each conditions
    **kwargs: further arguments for matplotlib.boxplot
        
    Returns
    -------
    result: dict
        The same as the return of matplotlib.boxplot
        See: .. plot:: mpl_examples/statistics/boxplot_demo.py

    Example
    -------
    data1 = [np.random.rand(50), np.random.rand(30)-.2, np.random.rand(10)+.3]
    data2 = [np.random.rand(40), np.random.rand(10)-.2, np.random.rand(15)+.3]

    plt.subplot(1,2,1)
    boxgroup([data1, data2], labels=("G1","G2","G3"), conditions=["C1","C2"])
    plt.subplot(1,2,2)
    boxgroup(data1, conditions=("G1","G2","G3"))
    """
    box_data = []
    cond_num = len(x)
    cond_loc = np.zeros(len(x))
    x_loc = np.array([])
    for i in range(len(x)):
        if type(x[i]) == np.ndarray:
            if len(x[i].shape) == 1:
                temp_loc = np.arange(1)
                box_data.append(x[i])
                group_num = 1
            else:
                group_num = x[i].shape[1]
                temp_loc = np.arange(x[i].shape[1])
                for j in range(x[i].shape[1]):
                    box_data.append(x[i][:,j])
        else:
            box_data += x[i]
            group_num = len(x[i])
            temp_loc = np.arange(len(x[i]))
        if i == 0:
            x_loc = temp_loc
            cond_loc[i] = np.mean(temp_loc)
        else:
            cond_loc[i] = np.mean(temp_loc)+x_loc[-1]+2
            x_loc = np.append(x_loc, x_loc[-1]+2+temp_loc)
        
    bp = plt.boxplot(box_data, notch, sys, positions=x_loc, widths=widths, 
                    patch_artist=patch_artist, **kwargs)
    if colors is None:
        colors = favorite_colors
    
    for i in range(len(box_data)):
        bp['medians'][i].set(color='blue', linewidth=2, alpha=alpha)
        bp['caps'][i].set(color='grey', linewidth=0, alpha=alpha)
        bp['caps'][i+len(box_data)].set(color='grey', linewidth=0, 
            alpha=alpha)
        bp['whiskers'][i].set(linestyle='solid', color='grey', 
            linewidth=2, alpha=alpha)
        bp['whiskers'][i+len(box_data)].set(linestyle='solid', 
            color='grey', linewidth=2, alpha=alpha)
        bp['boxes'][i].set(color=colors[i%group_num], 
            linewidth=2, alpha=alpha)
        
        if showmeans is True:
            plt.plot([x_loc[i]], [np.mean(box_data[i])],
                     color='w', marker='*', markersize=9)#, markeredgecolor='k'
        
    if labels is not None:
        for i in range(group_num):
            plt.scatter([], [], s=150, c=colors[i], marker='s', 
                        edgecolor='none', alpha=alpha, label=labels[i])
            
    if conditions is not None:
        plt.xticks(cond_loc, conditions)
    
    plt.xlim(x_loc[0]-0.7, x_loc[-1]+0.7)
    plt.legend(loc="best", scatterpoints=1, fancybox=True, ncol=group_num)
    plt.grid(alpha=0.4)

    return bp
