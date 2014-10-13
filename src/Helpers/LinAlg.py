__author__ = 'agross'

import pandas as pd
import numpy as np


def frame_svd(data_frame, impute='mean'):
    """
    Wrapper for taking in a pandas DataFrame, preforming SVD
    and outputting the U, S, and vH matricies in DataFrame form.
    """
    if impute == 'mean':
        data_frame = data_frame.dropna(thresh=int(data_frame.shape[1] * .75))
        data_frame = data_frame.fillna(data_frame.mean())

    U, S, vH = np.linalg.svd(data_frame.as_matrix(), full_matrices=False)
    U = pd.DataFrame(U, index=data_frame.index)
    vH = pd.DataFrame(vH, columns=data_frame.columns).T
    return U, S, vH


def extract_pc(df, pc_threshold=.2, standardize=True):
    """
    Wrapper for getting the first principal component from a DataFrame.
    Similar to frame_svd, but normalizes the data first and returns values in
    a different form.
    """
    if standardize:
        df = ((df.T - df.mean(1)) / df.std(1)).T
    try:
        U, S, vH = frame_svd(df)
    except np.linalg.LinAlgError:
        return None
    p = S ** 2 / sum(S ** 2)
    pat_vec = vH[0]
    gene_vec = U[0]
    pct_var = p[0]
    if sum(gene_vec) < 0:
        gene_vec = -1 * gene_vec
        pat_vec = -1 * pat_vec
    ret = {'pat_vec': pat_vec, 'gene_vec': gene_vec, 'pct_var': pct_var}
    return  ret if pct_var > pc_threshold else None


def drop_first_norm_pc(data_frame):
    """
    Normalize the data_frame by rows and then reconstruct it without the first
    principal component.  (Idea is to drop the biggest global pattern.)
    """
    norm = ((data_frame.T - data_frame.mean(1)) / data_frame.std(1)).T
    U, S, vH = frame_svd(norm)
    S[0] = 0  # zero out first pc
    rest = U.dot(pd.DataFrame(np.diag(S)).dot(vH.T))
    return rest