import numpy as np
import pylab as pl



# codes for ggplot like backgroud
# http://messymind.net/making-matplotlib-look-like-ggplot/
# Or simply import seaborn

favorite_colors=["deepskyblue", "limegreen", "orangered", "cyan", "magenta", 
                 "gold", "blueviolet", "dodgerblue", "greenyellow", "tomato",
                 "turquoise", "orchid", "darkorange", "mediumslateblue"]

def set_colors():
    """
    Set the favorite colors in matplotlib color_cycle and return the 
    list of favorite colors.
    """
    import matplotlib
    matplotlib.rcParams['axes.color_cycle'] = favorite_colors
    
    return favorite_colors


def venn3_plot(sets, set_labels=('A', 'B', 'C'), 
    set_colors=None, alpha=1.0, circle_on=False):
    """
    venn3 plot based on venn3 and venn3_circles from matplotlib_venn.

    Example:
    --------
    set1 = set(['A', 'B', 'C', 'D'])
    set2 = set(['B', 'C', 'D', 'E'])
    set3 = set(['C', 'D',' E', 'F', 'G'])
    venn3_plot([set1, set2, set3], ('Set1', 'Set2', 'Set3'))
    """
    from matplotlib_venn import venn3, venn3_circles

    if circle_on:
        v = venn3_circles(subsets=(1,1,1,1,1,1,1), alpha=0.8, color="r")
    if set_colors is None: 
        set_colors = favorite_colors[:3]
    v = venn3(subsets=(1,1,1,1,1,1,1), set_labels=set_labels, 
        set_colors=set_colors, alpha=alpha)
    v.get_label_by_id('111').set_text(len(sets[0]&sets[1]&sets[2]))
    v.get_label_by_id('110').set_text(len(sets[0]&sets[1]-sets[2]))
    v.get_label_by_id('101').set_text(len(sets[0]-sets[1]&sets[2]))
    v.get_label_by_id('100').set_text(len(sets[0]-sets[1]-sets[2]))
    v.get_label_by_id('011').set_text(len(sets[2]&sets[1]-sets[0]))
    v.get_label_by_id('010').set_text(len(sets[1]-sets[2]-sets[0]))
    v.get_label_by_id('001').set_text(len(sets[2]-sets[1]-sets[0]))

    return v


def boxgroup(x, labels=None, conditions=None, colors=None, notch=False, sys='',
             widths=1.0, patch_artist=True, showmeans=True, alpha=1, **kwargs):
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

    pl.subplot(1,2,1)
    boxgroup([data1, data2], labels=("G1","G2","G3"), conditions=["C1","C2"])
    pl.subplot(1,2,2)
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
        
    bp = pl.boxplot(box_data, notch, sys, positions=x_loc, widths=widths, 
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
            pl.plot([x_loc[i]], [np.mean(box_data[i])],
                    color='w', marker='*', markersize=9)#, markeredgecolor='k'
        
    if labels is not None:
        for i in range(group_num):
            pl.scatter([], [], s=150, c=colors[i], marker='s', 
                       edgecolor='none', alpha=alpha, label=labels[i])
            
    if conditions is not None:
        pl.xticks(cond_loc, conditions)
    
    pl.xlim(x_loc[0]-0.7, x_loc[-1]+0.7)
    pl.legend(loc="best", scatterpoints=1, fancybox=True, ncol=3)

    return bp
