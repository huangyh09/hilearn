# some wrapped functions from seaborn
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def regplot(x, y, hue=None, hue_values=None, show_corr=True, legend_on=True,
            **kwargs):
    """Extended plotting of `seaborn.regplot` with showing correlation 
    coeffecient and supporting multiple regression lines by hue (and hue_values) 

    Parameters
    ----------
    x: `array_like`, (1, ) 
        Values on x-axis
    y: `array_like`, (1, ) 
        Values on y-axis
    hue: `array_like`, (1, )
        Values to stratify samples into different groups
    hue_values: list or `array_like`
        A list of unique hue values; orders are retained in plotting layers
    show_corr: bool
        Whether show Pearson's correlation coefficient in legend
    legend_on: bool
        Whether display legend
    **kwargs: 
        for `seaborn.regplot`
        https://seaborn.pydata.org/generated/seaborn.regplot.html

    Returns
    -------
    ax: matplotlib Axes
        The Axes object containing the plot.
        same as seaborn.regplot

    Examples
    --------

    .. plot::

        >>> import numpy as np
        >>> from hilearn.plot import regplot
        >>> np.random.seed(1)
        >>> x1 = np.random.rand(50)
        >>> x2 = np.random.rand(50)
        >>> y1 = 2 * x1 + (0.5 + 2 * x1) * np.random.rand(50)
        >>> y2 = 4 * x2 + ((2 + x2) ** 2) * np.random.rand(50)
        >>> x, y = np.append(x1, x2), np.append(y1, y2)
        >>> hue = np.array(['group1'] * 50 + ['group2'] * 50)
        >>> regplot(x, y, hue)
    """
    import seaborn
    if hue is None:
        if show_corr:
            _label = "R=%.2f" %(st.pearsonr(x, y)[0])
        else:
            _label = None
        ax = seaborn.regplot(x, y, label=_label, **kwargs)
    else:
        if hue_values is None:
            hue_values = np.unique(hue)
        for hue_val in hue_values:
            _idx = hue == hue_val
            if show_corr:
                _label = str(hue_val) + ": R=%.2f" %(
                    st.pearsonr(x[_idx], y[_idx])[0])
            else:
                _label = None
            ax = seaborn.regplot(x[_idx], y[_idx], label=_label, **kwargs)
    
    if legend_on:
        plt.legend()
    
    return ax