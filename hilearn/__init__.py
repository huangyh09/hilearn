# Copyright(c) 2019, The HiLearn developer (Yuanhua Huang)

__version__ = "0.0.2"

from .base import match, id_mapping
from .stats import permutation_test
from .cross_validation import CrossValidation
from .plot.base_plot import boxgroup, venn3_plot
from .plot.corr_plot import corr_plot, ROC_plot, PR_curve
from .mixture_model.mix_linear_regression import MixLinearRegression


