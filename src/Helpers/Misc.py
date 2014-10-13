__author__ = 'agross'

import os as os
import pickle as pickle

import numpy as np
import scipy as sp
import pandas as pd


import matplotlib.pyplot as plt


def cluster_down(df, agg_function, dist_metric='euclidean', num_clusters=50,
                 draw_dendrogram=False):
    """
    Takes a DataFrame and uses hierarchical clustering to group along the index.
    Then aggregates the data in each group using agg_function to produce a matrix
    of prototypes representing each cluster.
    """
    d = sp.spacial.distance.pdist(df.as_matrix(), metric=dist_metric)
    D = sp.spacial.distance.squareform(d)
    Y = sp.cluster.hierarchy.linkage(D, method='complete')
    c = sp.cluster.hierarchy.fcluster(Y, num_clusters, criterion='maxclust')
    c = pd.Series(c, index=df.index, name='cluster')
    clustered = df.join(c).groupby('cluster').aggregate(agg_function)
    if draw_dendrogram:
        fig, ax = plt.subplots(1, 1, figsize=(14, 2))
        sp.clusterhierarchy.dendrogram(Y, color_threshold=np.sort(Y[:, 2])[-50],
                                       no_labels=True, count_sort='descendent')
        ax.set_frame_on(True)
        ax.set_yticks([])
        return clustered, c, fig
    return clustered, c


def get_random_genes(bp, lengths):
    """
    Use at own risk.
    """
    s = 0
    genes = []
    new_gene = 0
    while s < (bp + new_gene / 2.):
        i = np.random.random_integers(0, len(lengths) - 1)
        genes.append(i)
        new_gene = lengths.ix[i]
        s += new_gene
    genes = lengths.index[genes]
    return genes


def do_perm(f, vec, hit_mat, bp, lengths, iterations):
    """
    Use at own risk.
    """
    real_val = f(vec > 0)
    results = []
    for i in range(iterations):
        perm = hit_mat.ix[get_random_genes(bp, lengths)].sum() > 0
        results.append(f(perm))
    return sum(np.array(results) < real_val) / float(len(results))


def run_rate_permutation(df, hit_mat, gene_sets, lengths, f):
    """
    Use at own risk.
    """
    res = {}
    for p, genes in gene_sets.iteritems():
        if p not in df.index:
            continue
        bp = lengths[lengths.index.isin(genes)].sum()
        iterations = 10
        res[p] = do_perm(f, df.ix[p], hit_mat, bp, lengths, iterations)
        while (res[p] <= (10. / iterations)) and (iterations <= 2500):
            res[p] = do_perm(f, df.ix[p], hit_mat, bp, lengths, iterations)
            iterations = iterations * 5
    res = np.sort(pd.Series(res))
    return res


def make_path_dump(obj, file_path):
    """
    Pickle an object and save it in the given directory, if the directory
    not exist it is created.
    """
    dir_path = '/'.join(file_path.split('/')[:-1])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    pickle.dump(obj, open(file_path, 'wb'))


def merge_redundant(df):
    d = df.sort(axis=1).duplicated()
    features = {n: [n] for n, b in d.iteritems() if b == False}
    place = d.index[0]
    for idx, b in d.iteritems():
        if b == True:
            features[place] = features[place] + [idx]
        else:
            place = idx
    features = pd.Series(features)

    df = df.ix[d == False]
    df = df.rename(index=features.map(lambda s: '/'.join(s)))
    return df