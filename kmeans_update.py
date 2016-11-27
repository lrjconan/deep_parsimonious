import tensorflow as tf


def pdist(x, y):
    """ Compute Pairwise (Squared Euclidean) Distance

    Input:
        x: embedding of size M x D
        y: embedding of size N x D

    Output:
        dist: pairwise distance of size M x N
    """

    x2 = tf.tile(tf.expand_dims(tf.reduce_sum(tf.square(x), 1), 1),
                 tf.pack([1, tf.shape(y)[0]]))
    y2 = tf.tile(tf.transpose(tf.expand_dims(tf.reduce_sum(
        tf.square(y), 1), 1)), tf.pack([tf.shape(x)[0], 1]))
    xy = tf.matmul(x, y, transpose_b=True)
    return x2 - 2 * xy + y2


def assign_label(label, x, cluster_center):
    """ Assign Labels

    Input:
        x: embedding of size N x D
        label: cluster label of size N X 1
        K: number of clusters
        tf_eps: small constant

    Output:
        cluster_center: cluster center of size K x D
    """

    dist = pdist(x, cluster_center)
    return label.assign(tf.argmin(dist, 1))


def compute_mean(cluster_center, x, label, K, eta):
    """ Compute Mean

    Input:
        x: embedding of size N x D
        label: cluster label of size N X 1
        K: number of clusters
        tf_eps: small constant

    Output:
        cluster_center: cluster center of size K x D
    """
    tf_eps = tf.constant(1.0e-16)
    cluster_size = tf.expand_dims(tf.unsorted_segment_sum(
        tf.ones(label.get_shape()), label, K), 1)
    cluster_center_new = (1 - eta) * tf.unsorted_segment_sum(x,
                                                             label, K) / (cluster_size + tf_eps) + eta * cluster_center
    return cluster_center.assign(cluster_center_new)


def kmeans_clustering(x, cluster_center, label, K, eta):
    """ Spatial Clustering

    Input:
        x: embedding of size N x D
        cluster_center: cluster center of size K x D
        label: cluster label of size N X 1
        K: number of clusters
        eta: weight of moving average 

    Output:
        cluster_obj: objective function of clustering
    """

    label_op = assign_label(label, x, cluster_center)

    with tf.control_dependencies([label_op]):
        center_op = compute_mean(cluster_center, x, label, K, eta)

    do_updates = tf.group(label_op, center_op)

    return do_updates
