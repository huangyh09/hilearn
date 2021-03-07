# Copyright(c) 2019, The HiLearn developer (Yuanhua Huang)
from .version import __version__
from .utils.base import match, id_mapping
from .stats.base_stats import permutation_test
from .models.cross_validation import CrossValidation
from .models.mix_linear_regression import MixLinearRegression

from .plot import *


