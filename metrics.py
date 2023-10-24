"""
This file contains functions for evaluation metrics

"""
from sklearn.cluster import KMeans
import utils
import numpy as np
import sklearn.manifold as manifold
import matplotlib.pyplot as plt

def compute_silhouette_score(X, kmax):
    """
    used for the Elbow Method
    compute the silhouette for different values of k and chose the k
    the lower the better
    """
    from sklearn.metrics import silhouette_score
    sil = []
    # kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k, max_iter=1000).fit(X)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric = 'euclidean'))
    return sil


def calculate_WSS(points, kmax):
    """
    Used for the Elbow Method
    compute the Within-Cluster-Sum of Squared Errors(WSS) for different values of k and chose the k
    for which WSS becomes first stars to diminish.
    Returns:
            sse: the WSS score for points
    """
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)

    return sse


def visulize_tsne(X, perplextiy=[5, 30, 50, 100]):
    """
    Plots the tsne for the input data for different perplexities
    Args: X: the input data in numpy array format
            perplexity: list containing the different perplexities for the tsne graph
    returns:
    """
    n, d = X.shape

    y = 2
    red = y == 0
    green = y == 1

    (fig, subplots) = plt.subplots(4, figsize=(15, 8))

    perplextiy = perplextiy
    for i, perplextiy in enumerate(perplextiy):
        ax = subplots[i]

        tsne = manifold.TSNE(
            n_components=2,
            init='random',
            random_state=0,
            perplexity=perplextiy,
            learning_rate='auto',
            n_iter=5000
        )
        Y = tsne.fit_transform(X)
        ax.scatter(Y[:, 0], Y[:, 1])
        # ax.scatter(Y[red, 0], Y[red, 1], c="r")
        # ax.scatter(Y[green, 0], Y[green, 1], c="g")

    plt.show()


def plot_metrics(X, kmax):
    """
    Saves the WSS and silhouette score for k up to kmax
    Args: X: input data in numpy array
            kmax: maximum k to try for k-clustering
    returns:
    """
    n, d = X.shape
    WSS = calculate_WSS(X, d)
    sil = compute_silhouette_score(X, d)

    (fig, subplots) = plt.subplots(2, figsize=(15, 8))
    ax_WSS = subplots[0]
    ax_WSS.plot(np.arange(1, kmax + 1, 1), WSS)
    ax_WSS.set_title('WSS')
    ax_WSS.set_xlabel('k')
    ax_WSS.set_ylabel('WSS score')

    ax_sil = subplots[1]
    ax_sil.plot(np.arange(1, kmax, 1), sil)
    ax_sil.set_title('silhouette')
    ax_sil.set_xlabel('k')
    ax_sil.set_ylabel('silhouetee score')

    plt.show()

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not utilsis_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}