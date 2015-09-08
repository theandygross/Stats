'''
Created on Jul 17, 2013

@author: agross
'''
import pandas as pd
import numpy as np

from Helpers.RPY2 import robjects, pandas2ri, rpackages

lm = robjects.r.lm
base = rpackages.importr('base')

robjects.r.options(warn= -1);
zz = robjects.r.file("all.Rout", open="wt")
robjects.r.sink(zz, type='message')


def sanitize_for_r(f):
    s = f.name
    s = s.replace(':', '_').replace('(', '_').replace(')', '_').replace('-', '_')
    return pd.Series(f, name=s)


def process_factors(cov, standardize=True):
    """
    Coerce covariates and feature into format suitable for R's
    regression functions.
    """
    if type(cov) == pd.Series:
        cov = pd.concat([cov], axis=1)
    elif type(cov) == list:
        assert map(type, cov) == ([pd.Series] * len(cov))
        cov = pd.concat(cov, axis=1).dropna()

    c_real = cov.ix[:, cov.dtypes.isin([np.dtype(float), np.dtype(int),
                                            pd.np.dtype('float64')])]
    if standardize:
        c_real = (c_real - c_real.mean()) / c_real.std()
    df = c_real.combine_first(cov)
    df = df.groupby(level=0).first()
    df = df.dropna()
    df = pandas2ri.py2ri(df)
    return df


def stratified_regression(target, feature, strata):
    target, feature, strata = map(sanitize_for_r, [target, feature, strata])
    fmla = '{} ~ {} + strata({})'.format(target.name, feature.name, strata.name)
    fmla = robjects.Formula(fmla)
    df_r = process_factors([strata, target, feature])
    fit = lm(fmla, df_r)
    
    fmla = '{} ~ strata({})'.format(target.name, strata.name)
    fmla = robjects.Formula(fmla)
    fit_null = lm(fmla, df_r)
    
    f_stat = robjects.r.anova(fit_null, fit)[4][1]
    p = robjects.r.anova(fit_null, fit)[5][1]
    
    return pd.Series({'F': f_stat, 'p': p})