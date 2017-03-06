# Copyright(c) 2016, The HiLearn developer (Yuanhua Huang)
# Licensed under the MIT License at
# http://opensource.org/licenses/MIT

__version__ = "0.0.1"


from .base import id_mapping
from .stats import permutation_test
from .cross_validation import CrossValidation
from .plot.base_plot import boxgroup, venn3_plot
from .plot.corr_plot import corr_plot, ROC_plot, PR_curve
from .mixture_model.mix_linear_regression import MixLinearRegression


