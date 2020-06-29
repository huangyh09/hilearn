# some wrapped functions from seaborn
import numpy as np
import scipy.stats as st

def regplot(x, y, hue=None, hue_values=None, show_corr=True, **kwargs):
    """Wrap plot of `seaborn.regplot` with supporting hue and showing 
    correlation coeffecient.

    Parameters
    ----------
    x: `array_like`, (1, ) for values on x-axis
    y: `array_like`, (1, ) for values on y-axis
    hue: `array_like`, (1, )
        Values to stratify samples into different groups
    hue_values: list or `array_like`
        A list of unique hue values; orders are kept in layers
    show_corr: bool
        Whether show Pearson's correlation coefficient in legend
    **kwargs: for `seaborn.regplot`
    """
    import seaborn
    if hue is None:
        if show_corr:
            _label = "R=%.2f" %(st.pearsonr(x, y)[0])
        else:
            _label = None
        seaborn.regplot(x, y, label=_label, **kwargs)
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
            seaborn.regplot(x[_idx], y[_idx], label=_label, **kwargs)
    