# These codes come from the following blog
# http://messymind.net/making-matplotlib-look-like-ggplot/


from matplotlib import *

def rstyle(ax):
    """Styles an axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been carried out (needs to know final tick spacing)
    """
    #set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.85')
    ax.set_axisbelow(True)
   
    # #set minor tick spacing to 1/2 of the major ticks
    # ax.xaxis.set_minor_locator(MultipleLocator( (plt.xticks()[0][1]-plt.xticks()[0][0]) / 2.0 ))
    # ax.yaxis.set_minor_locator(MultipleLocator( (plt.yticks()[0][1]-plt.yticks()[0][0]) / 2.0 ))
   
    #remove axis border
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)
       
    #restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)
   
    #remove the minor tick lines    
    for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
        line.set_markersize(0)
   
    #only show bottom left ticks, pointing out of axis
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
   
   
    if ax.legend_ <> None:
        lg = ax.legend_
        lg.get_frame().set_linewidth(0)
        lg.get_frame().set_alpha(0.5)


def venn3_plot(sets, set_labels=('A', 'B', 'C'), 
    set_colors=('orangered', 'lawngreen', 'dodgerblue'),
    alpha=0.6, circle_on=False):
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
    v = venn3(subsets=(1,1,1,1,1,1,1), set_labels=set_labels, 
        set_colors=set_colors, alpha=alpha)
    v.get_label_by_id('111').set_text(len(sets[0]&sets[1]&sets[2]))
    v.get_label_by_id('110').set_text(len(sets[0]&sets[1]-sets[2]))
    v.get_label_by_id('101').set_text(len(sets[0]-sets[1]&sets[2]))
    v.get_label_by_id('100').set_text(len(sets[0]-sets[1]-sets[2]))
    v.get_label_by_id('011').set_text(len(sets[2]&sets[1]-sets[0]))
    v.get_label_by_id('010').set_text(len(sets[1]-sets[2]-sets[0]))
    v.get_label_by_id('001').set_text(len(sets[2]-sets[1]-sets[0]))

