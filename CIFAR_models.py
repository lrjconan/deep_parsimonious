import tensorflow as tf
import nn_cell_lib as nn


def baseline_model(param):
    """ Build a Alex-net style model """

    ops = {}
    conv_filters = {'filter_shape': param[
        'filter_shape'], 'filter_stride': param['filter_stride']}
    pooling = {'func_name': param['pool_func'], 'pool_size': param[
        'pool_size'], 'pool_stride': param['pool_stride']}

    device = '/cpu:0'
    if 'device' in param.keys():
        device = param['device']

    with tf.device(device):
        input_images = tf.placeholder(tf.float32, [None, param['img_height'], param[
                                      'img_width'], param['img_channel']])
        input_labels = tf.placeholder(tf.int32, [None])

        ops['input_images'] = input_images
        ops['input_labels'] = input_labels

        # build a CNN
        CNN = nn.CNN(
            conv_filters,
            pooling,
            param['act_func_cnn'],
            init_std=param['init_std_cnn'],
            wd=param['weight_decay'],
            scope='CNN')

        # build a MLP
        MLP = nn.MLP(
            param['dims_mlp'],
            param['act_func_mlp'],
            init_std=param['init_std_mlp'],
            wd=param['weight_decay'],
            scope='MLP')

        # prediction model
        feat_map = CNN.run(input_images)
        faet_map_MLP = tf.reshape(feat_map[-1], [-1, param['dims_mlp'][-1]])
        logits = MLP.run(faet_map_MLP)[-1]
        scaled_logits = tf.nn.softmax(logits)
        ops['scaled_logits'] = scaled_logits

        # compute cross-entropy loss
        CE_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, input_labels))
        ops['CE_loss'] = CE_loss

        # setting optimization
        global_step = tf.Variable(0.0, trainable=False)
        learn_rate = tf.train.exponential_decay(param['base_learn_rate'], global_step, param[
                                                'learn_rate_decay_step'], param['learn_rate_decay_rate'], staircase=True)

        # plain optimizer
        ops['train_step'] = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=param[
                                                       'momentum']).minimize(CE_loss, global_step=global_step)

    return ops


def clustering_model(param):
    """ Build a Alex-net style model with clustering """

    ops = {}
    conv_filters = {'filter_shape': param[
        'filter_shape'], 'filter_stride': param['filter_stride']}
    pooling = {'func_name': param['pool_func'], 'pool_size': param[
        'pool_size'], 'pool_stride': param['pool_stride']}

    device = '/cpu:0'
    if 'device' in param.keys():
        device = param['device']

    num_layer_cnn = len(param['num_cluster_cnn'])
    num_layer_mlp = len(param['num_cluster_mlp'])

    with tf.device(device):
        input_images = tf.placeholder(tf.float32, [None, param['img_height'], param[
                                      'img_width'], param['img_channel']])
        input_labels = tf.placeholder(tf.int32, [None])
        input_eta = tf.placeholder(tf.float32, [])

        c_reset_idx_cnn = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_cnn)]
        s_reset_idx_cnn = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_cnn)]
        c_reset_idx_mlp = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_mlp)]
        s_reset_idx_mlp = [tf.placeholder(tf.int32, [None])
                           for _ in xrange(num_layer_mlp)]

        ops['input_images'] = input_images
        ops['input_labels'] = input_labels
        ops['input_eta'] = input_eta
        ops['c_reset_idx_cnn'] = c_reset_idx_cnn
        ops['s_reset_idx_cnn'] = s_reset_idx_cnn
        ops['c_reset_idx_mlp'] = c_reset_idx_mlp
        ops['s_reset_idx_mlp'] = s_reset_idx_mlp

        # build a CNN
        CNN = nn.CNN_cluster(
            conv_filters=conv_filters,
            pooling=pooling,
            clustering_type=param['clustering_type_cnn'],
            clustering_shape=param['clustering_shape_cnn'],
            alpha=param['clustering_alpha_cnn'],
            num_cluster=param['num_cluster_cnn'],
            activation=param['act_func_cnn'],
            wd=param['weight_decay'],
            init_std=param['init_std_cnn'],
            scope='my_CNN')

        # build a MLP
        MLP = nn.MLP_cluster(
            dims=param['dims_mlp'],
            clustering_shape=param['clustering_shape_mlp'],
            alpha=param['clustering_alpha_mlp'],
            num_cluster=param['num_cluster_mlp'],
            activation=param['act_func_mlp'],
            init_std=param['init_std_mlp'],
            scope='my_MLP')

        # prediction ops
        feat_map, embedding_cnn, clustering_ops_cnn, reg_ops_cnn, reset_ops_cnn = CNN.run(
            input_images, input_eta, c_reset_idx_cnn, s_reset_idx_cnn)

        feat_map_mlp = tf.reshape(feat_map[-1], [-1, param['dims_mlp'][-1]])

        logits, embedding_mlp, clustering_ops_mlp, reg_ops_mlp, reset_ops_mlp = MLP.run(
            feat_map_mlp, input_eta, c_reset_idx_mlp, s_reset_idx_mlp)

        logits = logits[-1]
        scaled_logits = tf.nn.softmax(logits)
        ops['scaled_logits'] = scaled_logits
        ops['cluster_label'] = []
        ops['cluster_center'] = []

        for ii, cc in enumerate(CNN.cluster_center):
            if cc is not None:
                ops['cluster_label'] += [CNN.cluster_label[ii]]
                ops['cluster_center'] += [cc]

        for ii, cc in enumerate(MLP.cluster_center):
            if cc is not None:
                ops['cluster_label'] += [MLP.cluster_label[ii]]
                ops['cluster_center'] += [cc]

        ops['embeddings'] = embedding_cnn + embedding_mlp
        ops['clustering_ops'] = clustering_ops_cnn + clustering_ops_mlp
        ops['reg_ops'] = reg_ops_cnn + reg_ops_mlp
        ops['reset_ops'] = reset_ops_cnn + reset_ops_mlp
        reg_term = tf.reduce_sum(tf.pack(ops['reg_ops']))

        # compute cross-entropy loss
        CE_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, input_labels))
        ops['CE_loss'] = CE_loss

        # setting optimization
        global_step = tf.Variable(0.0, trainable=False)
        learn_rate = tf.train.exponential_decay(param['base_learn_rate'], global_step, param[
                                                'learn_rate_decay_step'], param['learn_rate_decay_rate'], staircase=True)

        # plain optimizer
        ops['train_step'] = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=param[
                                                       'momentum']).minimize(CE_loss + reg_term, global_step=global_step)

    return ops
