__author__ = 'agross'

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri

pandas2ri.activate()

formula_r = robjects.Formula
binomial = robjects.r.binomial('logit')


def convert_to_r_series(s):
    s_r = robjects.FloatVector(s)
    s_r.name = list(s.index)
    return s_r


def r_cbind(df_r, s):
    """
    df_r is a rpy2 DataFrame
    s is a Pandas Series
    """
    idx = list(robjects.r.rownames(df_r))
    s_r = convert_to_r_series(s.ix[idx])
    df_r_new = df_r.cbind(s_r)
    df_r_new.names[-1] = s.name
    return df_r_new