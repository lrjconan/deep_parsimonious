"""
Basic cells of neural networks

Renjie Liao
"""

import numpy as np
import tensorflow as tf
from kmeans_update import kmeans_clustering


def weight_variable(shape, init_method=None, dtype=tf.float32, init_param=None, wd=None, name=None, trainable=True):
    """ Initialize Weights 

    Input:
        shape: list of int, shape of the weights
        init_method: string, indicates initialization method
        init_param: a dictionary, 
        init_val: if it is not None, it should be a tensor
        wd: a float, weight decay
        name:
        trainable:

    Output:
        var: a TensorFlow Variable
    """

    if init_method is None:
        initializer = tf.zeros_initializer(shape, dtype=dtype)
    elif init_method == 'normal':
        initializer = tf.random_normal_initializer(
            mean=init_param['mean'], stddev=init_param['stddev'], seed=1, dtype=dtype)
    elif init_method == 'truncated_normal':
        initializer = tf.truncated_normal_initializer(
            mean=init_param['mean'], stddev=init_param['stddev'], seed=1, dtype=dtype)
    elif init_method == 'uniform':
        initializer = tf.random_uniform_initializer(
            minval=init_param['minval'], maxval=init_param['maxval'], seed=1, dtype=dtype)
    elif init_method == 'constant':
        initializer = tf.constant_initializer(
            value=init_param['val'], dtype=dtype)
    else:
        raise ValueError('Non supported initialization method!')

    var = tf.Variable(initializer(shape), name=name, trainable=trainable)

    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_decay')
        tf.add_to_collection('weight_decay', weight_decay)

    return var


class MLP(object):
    """ Multi Layer Perceptron (MLP)
        Note: the number of layers is N

    Input:
        dims: a list of N+1 int, number of hidden units (last one is the input dimension)
        activation: a list of N activation function names
        add_bias: a boolean, indicates whether adding bias or not
        wd: a float, weight decay 
        init_weights: a list of dictionaries of tensors, a dictionary has keys ['w', 'b'] for one layer
        model: a dictionary, contains all variables for future use, e.g., debug
        scope: tf scope of the model

    Output:
        a function which outputs a list of N tensors, each is the hidden activation of one layer 
    """

    def __init__(self, dims, activation=None, add_bias=True, wd=None, init_weights=None, init_std=None, scope='MLP'):
        num_layer = len(dims) - 1
        self.num_layer = num_layer
        self.w = [None] * num_layer
        self.b = [None] * num_layer
        self.act_func = [None] * num_layer
        self.dims = dims
        self.activation = activation
        self.add_bias = add_bias
        self.wd = wd
        self.init_weights = init_weights
        self.init_std = init_std
        self.scope = scope

        # initialize variables
        with tf.variable_scope(scope):
            for ii in xrange(num_layer):
                with tf.variable_scope('layer_{}'.format(ii)):
                    dim_in = dims[ii - 1]
                    dim_out = dims[ii]

                    if init_weights and init_weights[ii] is not None:
                        self.w[ii] = init_weights[ii]['w']
                    else:
                        self.w[ii] = weight_variable([dim_in, dim_out], init_method='truncated_normal', init_param={
                                                     'mean': 0.0, 'stddev': init_std[ii]}, wd=wd, name='w')

                    print 'MLP weight size in layer {}: {}'.format(ii, [dim_in, dim_out])

                    if add_bias:
                        if init_weights and init_weights[ii] is not None:
                            self.b[ii] = init_weights[ii]['b']
                        else:
                            self.b[ii] = weight_variable([dim_out], init_method='constant', init_param={
                                                         'val': 0.0}, wd=wd, name='b')
                        print 'MLP bias size in layer {}: {}'.format(ii, dim_out)

                    if activation and activation[ii] is not None:
                        if activation[ii] == 'relu':
                            act_func[ii] = tf.nn.relu
                        elif activation[ii] == 'sigmoid':
                            act_func[ii] = tf.sigmoid
                        elif activation[ii] == 'tanh':
                            act_func[ii] = tf.tanh
                        else:
                            raise ValueError('Non supported activation method!')

                    print 'MLP activate function in layer {}: {}'.format(ii, activation[ii])

    def run(self, x):
        h = [None] * self.num_layer

        with tf.variable_scope(self.scope):
            for ii in xrange(self.num_layer):
                with tf.variable_scope('layer_{}'.format(ii)):
                    if ii == 0:
                        input_vec = x
                    else:
                        input_vec = h[ii - 1]

                    h[ii] = tf.matmul(input_vec, self.w[ii])

                    if self.add_bias:
                        h[ii] += self.b[ii]

                    if self.act_func and self.act_func[ii] is not None:
                        h[ii] = self.act_func[ii](h[ii])

        return h


class CNN(object):
    """ Convolutional Neural Network (CNN)
        Note: the number of layers is N
              each layer looks like 'conv + [relu] + [pool]', [] means optional

    Input:
        conv_filters: a dictionary
                      key 'filter_shape': a list of N lists, each is 4-d list (H, W, C_in, C_out) specify the shape of a filter
                      key 'filter_stride': a list of N lists, each is 4-d list (B, H, W, C) specify the stride of filters in one layer
        pooling: a dictionary
                      key 'func_name': a list of N strings, each is name of pooling method ['max', 'avg']
                      key 'pool_size': a list of N lists, each is 4-d list specify the size of pooling in one layer
                      key 'pool_stride': a list of N lists, each is 4-d list specify the stride of pooling in one layer
        activation: a list of N activation function names
        add_bias: a boolean, indicates whether adding bias or not
        wd: a float, weight decay 
        init_weights: a list of dictionaries, each dict has keys ['w', 'b'] for [weight, bias]
        model: a dictionary, contains all variables 
        scope: tf scope of the model

    Output:
        a function which outputs a list of N tensors, each is the feature map of one layer 
    """

    def __init__(self, conv_filters, pooling, activation=None, add_bias=True, wd=None, init_std=None, init_weights=None, scope='CNN'):
        num_layer = len(conv_filters['filter_shape'])
        self.num_layer = num_layer

        self.w = [None] * num_layer
        self.b = [None] * num_layer
        self.pool_func = [None] * num_layer
        self.act_func = [None] * num_layer
        self.conv_filters = conv_filters
        self.pooling = pooling
        self.add_bias = add_bias
        self.init_std = init_std
        self.init_weights = init_weights
        self.scope = scope

        print 'CNN: {}'.format(scope)
        print 'Activation: {}'.format(activation)

        with tf.variable_scope(scope):
            for ii in xrange(num_layer):
                with tf.variable_scope('layer_{}'.format(ii)):
                    if init_weights and init_weights[ii] is not None:
                        self.w[ii] = init_weights[ii]['w']
                    else:
                        self.w[ii] = weight_variable(conv_filters['filter_shape'][ii], init_method='truncated_normal', init_param={
                            'mean': 0.0, 'stddev': init_std[ii]}, wd=wd, name='w')

                    print 'CNN filter size of layer {}: {}'.format(ii, conv_filters['filter_shape'][ii])

                    if add_bias:
                        if init_weights and init_weights[ii] is not None:
                            self.b[ii] = init_weights[ii]['b']
                        else:
                            self.b[ii] = weight_variable([conv_filters['filter_shape'][ii][3]], init_method='constant', init_param={
                                'val': 0}, wd=wd, name='b')

                        print 'CNN bias size in layer {}: {}'.format(ii, conv_filters['filter_shape'][ii][3])

                    if pooling['func_name'] and pooling['func_name'][ii] is not None:
                        if pooling['func_name'][ii] == 'max':
                            self.pool_func[ii] = tf.nn.max_pool
                        elif pooling['func_name'][ii] == 'avg':
                            self.pool_func[ii] = tf.nn.avg_pool
                        else:
                            raise ValueError('Non supported pooling method!')

                    if activation and activation[ii] is not None:
                        if activation[ii] == 'relu':
                            self.act_func[ii] = tf.nn.relu
                        elif activation[ii] == 'sigmoid':
                            self.act_func[ii] = tf.sigmoid
                        elif activation[ii] == 'tanh':
                            self.act_func[ii] = tf.tanh
                        else:
                            raise ValueError('Non supported activation method!')

    def run(self, x):
        """ x must be of size [B H W C] """
        h = [None] * self.num_layer

        with tf.variable_scope(self.scope):
            for ii in xrange(self.num_layer):
                if ii == 0:
                    input_vec = x
                else:
                    input_vec = h[ii - 1]

                h[ii] = tf.nn.conv2d(input_vec, self.w[ii], self.conv_filters[
                                     'filter_stride'][ii], padding='SAME')

                if self.add_bias:
                    h[ii] += self.b[ii]

                if self.act_func[ii] is not None:
                    h[ii] = self.act_func[ii](h[ii])

                if self.pool_func[ii] is not None:
                    h[ii] = self.pool_func[ii](h[ii], ksize=self.pooling['pool_size'][
                                               ii], strides=self.pooling['pool_stride'][ii], padding='SAME')

        return h


class CNN_cluster(object):
    """ Convolutional Neural Network (CNN) with Clustering
        Note: the number of layers is N
              each layer looks like 'conv + [relu] + [pool]', [] means optional

    Input:
        conv_filters: a dictionary
            key 'filter_shape': a list of N lists, each is 4-d list (H, W, C_in, C_out) specify the shape of a filter
            key 'filter_stride': a list of N lists, each is 4-d list (B, H, W, C) specify the stride of filters in one layer
        pooling: a dictionary
            key 'func_name': a list of N strings, each is name of pooling method ['max', 'avg']
            key 'pool_size': a list of N lists, each is 4-d list specify the size of pooling in one layer
            key 'pool_stride': a list of N lists, each is 4-d list specify the stride of pooling in one layer
        clustering_type: list of string, size N, {'sample', 'spatial', 'channel'} 
        clustering_shape: list of lists, size M X D, M = number of clusters, D = dimension of cluster
        activation: a list of N activation function names
        add_bias: a boolean, indicates whether adding bias or not
        wd: a float, weight decay 
        init_weights: a list of dictionaries, each dict has keys ['w', 'b'] for [weight, bias]
        model: a dictionary, contains all variables 
        scope: tf scope of the model

    Output:
        a function which outputs a list of N tensors, each is the feature map of one layer 
    """

    def __init__(self, conv_filters, pooling, clustering_type, clustering_shape, alpha, num_cluster, activation=None, add_bias=True, wd=None, init_std=None, init_weights=None, scope='CNN_cluster'):

        num_layer = len(conv_filters['filter_shape'])
        self.num_layer = num_layer
        self.w = [None] * num_layer
        self.b = [None] * num_layer
        self.pool_func = [None] * num_layer
        self.act_func = [None] * num_layer
        self.cluster_center = [None] * num_layer
        self.cluster_label = [None] * num_layer
        self.add_bias = add_bias
        self.scope = scope
        self.conv_filters = conv_filters
        self.pooling = pooling
        self.clustering_type = clustering_type
        self.clustering_shape = clustering_shape
        self.alpha = alpha
        self.num_cluster = num_cluster

        print 'CNN: {}'.format(scope)
        print 'Activation: {}'.format(activation)

        with tf.variable_scope(scope):
            for ii in xrange(num_layer):
                with tf.variable_scope('layer_{}'.format(ii)):
                    if init_weights and init_weights[ii] is not None:
                        self.w[ii] = init_weights[ii]['w']
                    else:
                        self.w[ii] = weight_variable(conv_filters['filter_shape'][ii], init_method='truncated_normal', init_param={
                                                     'mean': 0.0, 'stddev': init_std[ii]}, wd=wd, name='w')

                    print 'CNN filter size in layer {}: {}'.format(ii, conv_filters['filter_shape'][ii])

                    if clustering_shape[ii]:
                        self.cluster_center[ii] = weight_variable(
                            [num_cluster[ii], clustering_shape[ii][1]],
                            init_method='truncated_normal',
                            init_param={'mean': 0.0, 'stddev': init_std[ii]},
                            name='cluster_center', trainable=False)

                        if clustering_shape[ii][0] < num_cluster[ii]:
                            random_init_label = np.random.choice(
                                num_cluster[ii], clustering_shape[ii][0], replace=False)
                        else:
                            random_init_label = np.concatenate([np.random.permutation(num_cluster[ii]), np.random.choice(
                                num_cluster[ii], clustering_shape[ii][0] - num_cluster[ii])])

                        self.cluster_label[ii] = tf.Variable(
                            random_init_label, name='cluster_label', trainable=False, dtype=tf.int64)

                    if add_bias:
                        if init_weights and init_weights[ii] is not None:
                            self.b[ii] = init_weights[ii]['b']
                        else:
                            self.b[ii] = weight_variable([conv_filters['filter_shape'][ii][
                                                         3]], init_method='constant', init_param={'val': 0.0}, wd=wd, name='b')

                        print 'CNN filter bias in layer {}: {}'.format(ii, conv_filters['filter_shape'][ii][3])

                    if pooling['func_name'] and pooling['func_name'][ii] is not None:
                        if pooling['func_name'][ii] == 'max':
                            self.pool_func[ii] = tf.nn.max_pool
                        elif pooling['func_name'][ii] == 'avg':
                            self.pool_func[ii] = tf.nn.avg_pool
                        else:
                            raise ValueError('Unsupported pooling method!')

                    if activation and activation[ii] is not None:
                        if activation[ii] == 'relu':
                            self.act_func[ii] = tf.nn.relu
                        elif activation[ii] == 'sigmoid':
                            self.act_func[ii] = tf.sigmoid
                        elif activation[ii] == 'tanh':
                            self.act_func[ii] = tf.tanh
                        else:
                            raise ValueError('Unsupported activation method!')

    def run(self, x, eta, idx_center=None, idx_sample=None):
        """ x must be of size [B H W C] """
        h = [None] * self.num_layer
        embeddings = []
        reg_ops = []
        reset_ops = []
        clustering_ops = []

        with tf.variable_scope(self.scope):
            for ii in xrange(self.num_layer):
                if ii == 0:
                    input_vec = x
                else:
                    input_vec = h[ii - 1]

                h[ii] = tf.nn.conv2d(input_vec, self.w[ii], self.conv_filters[
                                     'filter_stride'][ii], padding='SAME')

                if self.add_bias:
                    h[ii] += self.b[ii]

                if self.clustering_type[ii] == 'sample':
                    embedding = h[ii]
                elif self.clustering_type[ii] == 'spatial':
                    embedding = h[ii]
                elif self.clustering_type[ii] == 'channel':
                    embedding = tf.transpose(h[ii], [0, 3, 1, 2])

                if self.clustering_shape[ii] is not None:
                    embedding = tf.reshape(
                        embedding, [-1, self.clustering_shape[ii][1]])

                    embeddings += [embedding]
                    clustering_ops += [kmeans_clustering(embedding, self.cluster_center[
                                                         ii], self.cluster_label[ii], self.num_cluster[ii], eta)]

                    sample_center = tf.stop_gradient(
                        tf.gather(self.cluster_center[ii], self.cluster_label[ii]))
                    reg_ops += [tf.reduce_mean(tf.square(embedding -
                                                         sample_center)) * self.alpha[ii] / 2.0]

                    reset_ops += [tf.scatter_update(self.cluster_center[ii], idx_center[
                        ii], tf.gather(embedding, idx_sample[ii]))]

                if self.act_func[ii] is not None:
                    h[ii] = self.act_func[ii](h[ii])

                if self.pool_func[ii] is not None:
                    h[ii] = self.pool_func[ii](h[ii], ksize=self.pooling['pool_size'][
                                               ii], strides=self.pooling['pool_stride'][ii], padding='SAME')

        return h, embeddings, clustering_ops, reg_ops, reset_ops


class MLP_cluster(object):
    """ Multi Layer Perceptron (MLP)
        Note: the number of layers is N

    Input:
        dims: a list of N+1 int, number of hidden units (last one is the input dimension)
        activation: a list of N activation function names
        add_bias: a boolean, indicates whether adding bias or not
        wd: a float, weight decay 
        init_weights: a list of dictionaries of tensors, a dictionary has keys ['w', 'b'] for one layer
        model: a dictionary, contains all variables for future use, e.g., debug
        scope: tf scope of the model

    Output:
        a function which outputs a list of N tensors, each is the hidden activation of one layer 
    """

    def __init__(self, dims, clustering_shape, alpha, num_cluster, activation=None, add_bias=True, wd=None, init_weights=None, init_std=None, scope='MLP'):
        num_layer = len(dims) - 1
        self.num_layer = num_layer
        self.w = [None] * num_layer
        self.b = [None] * num_layer
        self.act_func = [None] * num_layer
        self.cluster_center = [None] * num_layer
        self.cluster_label = [None] * num_layer
        self.dims = dims
        self.activation = activation
        self.add_bias = add_bias
        self.wd = wd
        self.init_weights = init_weights
        self.init_std = init_std
        self.clustering_shape = clustering_shape
        self.alpha = alpha
        self.num_cluster = num_cluster
        self.scope = scope

        # initialize variables
        with tf.variable_scope(scope):
            for ii in xrange(num_layer):
                with tf.variable_scope('layer_{}'.format(ii)):
                    dim_in = dims[ii - 1]
                    dim_out = dims[ii]

                    if init_weights and init_weights[ii] is not None:
                        self.w[ii] = init_weights[ii]['w']
                    else:
                        self.w[ii] = weight_variable([dim_in, dim_out], init_method='truncated_normal', init_param={
                                                     'mean': 0.0, 'stddev': init_std[ii]}, wd=wd, name='w')

                    print 'MLP weight size in layer {}: {}'.format(ii, [dim_in, dim_out])

                    if clustering_shape[ii]:
                        self.cluster_center[ii] = weight_variable(
                            [num_cluster[ii], clustering_shape[ii][1]],
                            init_method='truncated_normal',
                            init_param={'mean': 0.0, 'stddev': init_std[ii]},
                            name='cluster_center', trainable=False)

                        if clustering_shape[ii][0] < num_cluster[ii]:
                            random_init_label = np.random.choice(
                                num_cluster[ii], clustering_shape[ii][0], replace=False)
                        else:
                            random_init_label = np.concatenate([np.random.permutation(num_cluster[ii]), np.random.choice(
                                num_cluster[ii], clustering_shape[ii][0] - num_cluster[ii])])

                        self.cluster_label[ii] = tf.Variable(
                            random_init_label, name='cluster_label', trainable=False, dtype=tf.int64)

                    if add_bias:
                        if init_weights and init_weights[ii] is not None:
                            self.b[ii] = init_weights[ii]['b']
                        else:
                            self.b[ii] = weight_variable([dim_out], init_method='constant', init_param={
                                                         'val': 0.0}, wd=wd, name='b')

                        print 'MLP bias size in layer {}: {}'.format(ii, dim_out)

                    if activation and activation[ii] is not None:
                        if activation[ii] == 'relu':
                            act_func[ii] = tf.nn.relu
                        elif activation[ii] == 'sigmoid':
                            act_func[ii] = tf.sigmoid
                        elif activation[ii] == 'tanh':
                            act_func[ii] = tf.tanh
                        else:
                            raise ValueError('Non supported activation method!')

                    print 'MLP activate function in layer {}: {}'.format(ii, activation[ii])

    def run(self, x, eta, idx_center=None, idx_sample=None):
        h = [None] * self.num_layer
        embeddings = []
        reg_ops = []
        reset_ops = []
        clustering_ops = []

        with tf.variable_scope(self.scope):
            for ii in xrange(self.num_layer):
                with tf.variable_scope('layer_{}'.format(ii)):
                    if ii == 0:
                        input_vec = x
                    else:
                        input_vec = h[ii - 1]

                    h[ii] = tf.matmul(input_vec, self.w[ii])

                    if self.add_bias:
                        h[ii] += self.b[ii]

                    if self.clustering_shape[ii] is not None:
                        embedding = h[ii]
                        embeddings += [embedding]

                        clustering_ops += [kmeans_clustering(embedding, self.cluster_center[
                                                             ii], self.cluster_label[ii], self.num_cluster[ii], eta)]

                        sample_center = tf.stop_gradient(
                            tf.gather(self.cluster_center[ii], self.cluster_label[ii]))
                        reg_ops += [tf.reduce_mean(
                            tf.square(embedding - sample_center)) * self.alpha[ii] / 2.0]

                        reset_ops += [tf.scatter_update(self.cluster_center[ii], idx_center[
                            ii], tf.gather(h[ii], idx_sample[ii]))]

                    if self.act_func and self.act_func[ii] is not None:
                        h[ii] = self.act_func[ii](h[ii])

        return h, embeddings, clustering_ops, reg_ops, reset_ops
