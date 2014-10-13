__author__ = 'agross'

import pandas as pd
import numpy as np
import itertools as it

from statsmodels.sandbox.stats import multicomp
from numpy.linalg import LinAlgError, svd


def powerset(iterable):
    """
    "http://docs.python.org/2/library/itertools.html#recipes"
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    """
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in
                                  range(len(s) + 1))


def binarize(f):
    """
    Binarize a continuous vector by minimizing the difference in
    variance between the two resulting groups.
    """
    f = f - f.mean()
    f2 = (f.order() ** 2)
    split = f.ix[(f2.cumsum() - (f2.sum() / 2.)).abs().idxmin()]
    return f > split


def transfer_index(source, target):
    """
    Transfer index of source Series to target Series.
    """
    return pd.Series(list(target), index=source.index)


def true_index(s):
    """
    Return indicies for which the variable is true.
    """
    return s[s].index

ti = true_index  # short form version of this very useful function.


def combine(a, b):
    """
    Combine two categorical features.
    """
    combo = (a * 1.).add(b * 2.)
    combo = combo.dropna()
    if not a.name:
        a.name = 'first'
    if not b.name:
        b.name = 'second'
    if a.name != b.name:
        combo = combo.map({0: 'neither', 1: a.name, 2: b.name, 3:'both'})
    else:
        combo = combo.map({0: 'neither', 1: 'first', 2: 'second', 3:'both'})
    return combo


def to_quants(vec, q=.25, std=None, labels=False):
    """
    Get quantiles of a numerical vector.
    """
    vec = (vec - vec.mean()) / vec.std()
    if q == .5:
        vec = (vec > 0).astype(int)
        if labels:
            vec = vec.map({0:'Bottom 50%', 1:'Top 50%'})
    elif std is None:
        vec = ((vec > vec.quantile(1 - q)).astype(int) -
               (vec <= vec.quantile(q)).astype(int)).astype(float)
        if labels:
            vec = vec.map({-1:'Bottom {}%'.format(int(q * 100)), 0:'Normal',
                           1:'Top {}%'.format(int(q * 100))})
    else:
        vec = (vec - vec.mean()) / vec.std()
        vec = (1.*(vec > std) - 1.*(vec <= (-1 * std)))
        if labels:
            vec = vec.map({-1: 'low', 0: 'normal', 1:'high'})
    return vec


def add_column_level(tab, arr, name):
    """
    Add a level to a DataFrames columns.
    """
    tab = tab.T
    tab[name] = arr
    tab = tab.set_index(name, append=True)
    tab.index = tab.index.swaplevel(0, 1)
    return tab.T


def bhCorrection(s, n=None):
    """
    Benjamini-Hochberg correction for a Series of p-values.
    """
    s = s.fillna(1.)
    if n > len(s):
        p_vals = list(s) + [1] * (n - len(s))
    else:
        p_vals = list(s)
    q = multicomp.multipletests(p_vals, method='fdr_bh')[1][:len(s)]
    q = pd.Series(q[:len(s)], s.index, name='p_adj')
    return q


def match_series(a, b):
    """
    Matches two series on shared data.
    """
    a, b = a.align(b, join='inner', copy=False)
    valid = pd.notnull(a) & pd.notnull(b)
    a = a[valid]
    if not a.index.is_unique:
        a = a.groupby(lambda s: s).first()  # some sort of duplicate index bug
    b = b[valid]
    if not b.index.is_unique:
        b = b.groupby(lambda s: s).first()
    return a, b


def split_a_by_b(a, b):
    """
    Splits Series a by the groups defined in Series b.
    """
    a, b = match_series(a, b)
    groups = [a[b == num] for num in set(b)]
    return groups


def screen_feature(vec, test, df, align=True):
    """
    Screens a Series for a statistical test across a data-frame.

    vec:   Series to be tested.
    test:  test to be run, should return a p-value labeled as 'p' in a Series with
           any other data.
    df:    DataFrame of features to test against.  Features should be on the index.
    align: Optional flag to algin the vec and df.  Can improve speed of operations
           by doing this on the DataFrame first rather than each row separately.
           Can result in bugs from time to time, but easy to detect as the function
           returns NAs.
    """
    if align:
        df, vec = df.align(vec, axis=1)
    s = pd.DataFrame({f: test(vec, feature) for f, feature in df.iterrows()}).T
    s['q'] = bhCorrection(s.p)
    s = s.sort(columns='p')
    return s


def df_to_binary_vec(df):
    """
    Try and turn a DataFrame into binary vector... use at own risk.
    """
    cutoff = np.sort(df.sum())[-int(df.sum(1).mean())]
    if (len(df) > 2) and (cutoff == 1.):
        cutoff = 2
    vec = (df.sum() >= cutoff).astype(int)
    return vec


def get_vec_type(vec):
    if vec.count() < 10:
        return
    elif vec.dtype in [float, int]:
        return 'real'
    vc = vec.value_counts()
    if len(vc) == 1 or vc.order().iloc[-2] <= 5:
        return
    elif len(vc) == 2:
        return 'boolean'
    elif vec.dtype == 'object':
        return 'categorical'





