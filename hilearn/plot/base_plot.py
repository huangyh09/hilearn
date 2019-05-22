import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# codes for ggplot like backgroud
# http://messymind.net/making-matplotlib-look-like-ggplot/
# Or simply import seaborn

favorite_colors=["deepskyblue", "limegreen", "orangered", "cyan", "magenta", 
                 "gold", "blueviolet", "dodgerblue", "greenyellow", "tomato",
                 "turquoise", "orchid", "darkorange", "mediumslateblue"]

WeiZhu_colors = ['#4796d7', '#f79e54', '#79a702', '#df5858', '#556cab', 
                 '#de7a1f', '#ffda5c', '#4b595c', '#6ab186', '#bddbcf', 
                 '#daad58', '#488a99', '#f79b78', '#ffba00']

#seaborn_colors = seaborn.color_palette("hls", 8)

def set_colors(color_list=WeiZhu_colors):
    """
    Set the favorite colors in matplotlib color_cycle and return the 
    list of favorite colors.
    """
    matplotlib.rcParams['axes.color_cycle'] = color_list
    return color_list


def set_frame(ax):
    """Example of setting the frame of the plot box.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(direction='out')
    return ax


def set_style(label_size=12, grid_alpha=0.4):
    """
    Set the figure style
    """
    if grid_alpha is not None and grid_alpha >= 0 and grid_alpha <= 1:
        matplotlib.rcParams['grid.alpha'] = grid_alpha
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size * 1.1
    matplotlib.rcParams['axes.labelsize'] = label_size * 1.1
    matplotlib.rcParams['axes.titlesize'] = label_size * 1.2
    matplotlib.rcParams['axes.titleweight'] = 'bold'


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


def contour2D(x, y, f, N=10, cmap="bwr", contourLine=True, optima=True, **kwargs):
    """
    Plot 2D contour.
    
    Parameters
    ----------
    x: array like (m1, )
        The first dimention
    y: array like (m2, )
        The Second dimention
    f: a function
        The funciton return f(x1, y1)
    N: int
        The number of contours
    contourLine: bool
        Turn the contour line
    optima: bool
        Turn on the optima
    **kwargs: further arguments for matplotlib.boxplot
        
    Returns
    -------
    result: contourf
        The same as the return of matplotlib.plot.contourf
        See: .. plot:: http://matplotlib.org/examples/pylab_examples/contourf_demo.html

    Example
    -------
    def f(x, y):
        return -x**2-y**2
    x = np.linspace(-0.5, 0.5, 100)
    y = np.linspace(-0.5, 0.5, 100)
    contour2D(x, y, f)
    """
    X,Y = np.meshgrid(x,y)
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = f(X[i,j],Y[i,j])
    idx = np.argmax(Z)

    cf = plt.contourf(X, Y, Z, N, cmap=cmap, **kwargs)
    if contourLine is True:
        C = plt.contour(X, Y, Z, N, alpha=0.7, colors="k", linewidth=0.5)
        plt.clabel(C, inline=1, fontsize=10)
    if optima is True:
        plt.scatter(X[idx/100, idx%100], Y[idx/100, idx%100], s=120, marker='*')
                    #, marker=(5,2))
    
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    return cf


