import math
import numpy as np
import tensorflow as tf
from scipy.spatial import distance


def kmeans_plus_plus(cluster_center, data):
    """ Kmeans++ style update """

    pdist = distance.cdist(cluster_center, data, 'sqeuclidean')
    nn_cluster_idx = np.argsort(pdist, axis=0)[0]
    prob = pdist[nn_cluster_idx, range(data.shape[0])]

    return prob / sum(prob)


def get_update_cluster_idx(cluster_center, data, mask_update):
    idx_center = []
    idx_sample = []

    # detect empty cluster center
    cluster_idx = 0
    for ii in xrange(len(mask_update)):
        if mask_update[ii] is None:
            idx_center += [[]]
            idx_sample += [[]]
            continue

        empty_idx = mask_update[ii] == False
        non_empty_idx = mask_update[ii] == True

        num_replace = min(len(np.nonzero(empty_idx)[0]), data[
                          cluster_idx].shape[0])
        update_idx = np.nonzero(empty_idx)[0][:num_replace]

        idx_center += [update_idx]
        mask_update[ii][update_idx] = True

        if len(np.nonzero(non_empty_idx)[0]) > 0:
            # kmeans++ style update
            prob = kmeans_plus_plus(cluster_center[cluster_idx][
                                    non_empty_idx, :], data[cluster_idx])
        else:
            prob = np.ones(data[cluster_idx].shape[0]) / \
                data[cluster_idx].shape[0]

        idx_sample += [np.random.choice(data[cluster_idx].shape[0],
                                        num_replace, replace=False, p=prob)]

        cluster_idx += 1

    return idx_center, idx_sample
