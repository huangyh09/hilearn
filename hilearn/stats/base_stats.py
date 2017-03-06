
import numpy as np

def logistic(x):
    """
    Logistic function, mapping (-inf, inf) to (0,1)

    Parameters
    ----------
    x: float, int, array, list
        input variable

    Returns
    -------
    val: float, int, array
        logistic(x)
    """
    return np.exp(x)/(1+np.exp(x))
    
def logit(x, minval=0.001):
    """
    Logit function, mapping (0,1) to (-inf, inf)

    Parameters
    ----------
    x: float, int, array, list
        input variable
    minval: float (optional, default=0.001)
        minimum value of input x, and maximum of 1-x

    Returns
    -------
    val: float, int, array
        logit(x)
    """
    if isinstance(x,  (list, tuple, np.ndarray)):
        x[1-x<minval] = 1-minval
        x[x<minval] = minval
    else:
        x = max(minval, x)
        x = min(1-minval, x)
    val = np.log(x/(1-x))
    return val

def uninormal_pdf(x, mu, sigma, log=False):
    """
    Probability density function of univariate normal distribution

    Parameters
    ----------
    x: float, int, array, list
        input variable
    mu: float
        mean of the normal distribution
    sigma: float
        standard divation of the normal distibution
    log: bool (optional)
        If True, return pdf at log scale

    Returns
    -------
    pdf: float, int, array
        Probability density of x
    """
    mu = np.float(mu)
    sigma = np.float(sigma)
    pdf = np.log(1/(sigma*np.sqrt(2*np.pi))) - 0.5*((x-mu)/sigma)**2
    if log is False:
        pdf = np.exp(pdf)
    return pdf


def permutation_test(x1, x2, times=1000, sides=2, metrics="mean", seed=None):
    """
    Permutation test: whether group x1 has equal [mean|median] as group x2

    Parameters
    ----------
    x1: array or list of int/float
        samples for variable x1
    x2: array or list of int/float
        samples for variable x2
    times: int
        number of iterations, better <= 10^5
    sides: int, 1 or 2
        one-sided p value or two-sided p-value
    metrics: str
        the metrics: mean, median
    seed: int or None
        the seed for permutation

    Returns
    -------
    (deta, pval): float, float
        the difference between two groups and the p value

    Examples
    --------
    >>>x1 = np.random.normal(0,1,100)
    >>>x2 = np.random.normal(1,2,100)
    >>>permutation_test(x1, x2)
    """

    x1 = np.array(x1)
    x2 = np.array(x2)
    x1 = x1[x1 == x1]
    x2 = x2[x2 == x2]

    if metrics == "median":
        diff_obs = np.median(x1) - np.median(x2)
    else:
        diff_obs = np.mean(x1) - np.mean(x2)

    if seed is not None:
        np.random.seed(seed)

    cnt = 0
    xx = np.append(x1, x2)
    diff_all = np.zeros(times)
    for i in range(times):
        # np.random.shuffle(xx)
        xx = np.random.permutation(xx)
        if metrics == "median":
            diff_all[i] = np.median(xx[:len(x1)]) - np.median(xx[len(x1):])
        else:
            diff_all[i]  = np.mean(xx[:len(x1)]) - np.mean(xx[len(x1):])

    if sides == 2:
        pval = np.mean(np.abs(diff_all) >= np.abs(diff_obs))
    else:
        pval = np.mean(diff_all >= np.abs(diff_obs))

    return diff_obs, pval
