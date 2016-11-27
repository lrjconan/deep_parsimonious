import numpy as np
import tensorflow as tf
from nn_cell_lib import weight_variable
from kmeans_update import kmeans_clustering


def get_conv_cluster_embedding(clustering_type, clustering_dim, h):

    if clustering_type == 'sample':
        embedding = h
    elif clustering_type == 'spatial':
        embedding = h
    elif clustering_type == 'channel':
        embedding = tf.transpose(h, [0, 3, 1, 2])

    return tf.reshape(embedding, [-1, clustering_dim])


def compute_reg(embedding, c_center, c_label, c_alpha):
    sample_center = tf.stop_gradient(tf.gather(c_center, c_label))
    reg = tf.reduce_mean(tf.square(embedding - sample_center)) * c_alpha / 2.0

    return reg


def build_AlexNet_CUB_clustering(param):
    """ Build AlexNet for CUB dataset 
        Note: if use caffe-compatible weights, the input should be BWHC
    """
    ops = {}
    num_layer = 8
    wd = None if param['weight_decay'] == 0 else param['weight_decay']
    f_shape = param['filter_shape']
    c_type = param['clustering_type']
    c_shape = param['clustering_shape']
    c_alpha = param['clustering_alpha']
    num_cluster = param['num_cluster']
    init_std = param['init_std']
    init_bias = param['init_bias']
    bat_size = param['bat_size']

    with tf.device(param['device']):
        input_images = tf.placeholder(tf.float32, [None, 227, 227, 3])
        input_labels = tf.placeholder(tf.int32, [None])
        dropout_rate = tf.placeholder(tf.float32, [])
        phase_train = tf.placeholder(tf.bool, [])
        input_eta = tf.placeholder(tf.float32, [])

        model_w1 = tf.placeholder(tf.float32, [11, 11, 3, 96])
        model_w2 = tf.placeholder(tf.float32, [5, 5, 48, 256])
        model_w3 = tf.placeholder(tf.float32, [3, 3, 256, 384])
        model_w4 = tf.placeholder(tf.float32, [3, 3, 192, 384])
        model_w5 = tf.placeholder(tf.float32, [3, 3, 192, 256])
        model_w6 = tf.placeholder(tf.float32, [9216, 4096])
        model_w7 = tf.placeholder(tf.float32, [4096, 4096])
        model_w8 = tf.placeholder(tf.float32, [4096, 200])

        model_b1 = tf.placeholder(tf.float32, [96])
        model_b2 = tf.placeholder(tf.float32, [256])
        model_b3 = tf.placeholder(tf.float32, [384])
        model_b4 = tf.placeholder(tf.float32, [384])
        model_b5 = tf.placeholder(tf.float32, [256])
        model_b6 = tf.placeholder(tf.float32, [4096])
        model_b7 = tf.placeholder(tf.float32, [4096])
        model_b8 = tf.placeholder(tf.float32, [200])

        c_reset_idx = [tf.placeholder(tf.int32, [None])
                       for _ in xrange(num_layer - 1)]
        s_reset_idx = [tf.placeholder(tf.int32, [None])
                       for _ in xrange(num_layer - 1)]

        ops['input_images'] = input_images
        ops['input_labels'] = input_labels
        ops['input_eta'] = input_eta
        ops['phase_train'] = phase_train
        ops['dropout_rate'] = dropout_rate
        ops['c_reset_idx'] = c_reset_idx
        ops['s_reset_idx'] = s_reset_idx

        ops['model_w1'] = model_w1
        ops['model_w2'] = model_w2
        ops['model_w3'] = model_w3
        ops['model_w4'] = model_w4
        ops['model_w5'] = model_w5
        ops['model_w6'] = model_w6
        ops['model_w7'] = model_w7
        ops['model_w8'] = model_w8

        ops['model_b1'] = model_b1
        ops['model_b2'] = model_b2
        ops['model_b3'] = model_b3
        ops['model_b4'] = model_b4
        ops['model_b5'] = model_b5
        ops['model_b6'] = model_b6
        ops['model_b7'] = model_b7
        ops['model_b8'] = model_b8

        # initialize weights
        w = [[] for _ in xrange(num_layer)]
        b = [[] for _ in xrange(num_layer)]

        # with tf.variable_scope('Alex_net'):
        # init from scratch
        for ii in xrange(num_layer):
            w[ii] = weight_variable(f_shape[ii], init_method='truncated_normal', init_param={
                                    'mean': 0.0, 'stddev': init_std[ii]}, wd=wd, name='w_{}'.format(ii + 1))

            b[ii] = weight_variable([f_shape[ii][-1]], init_method='constant', init_param={
                                    'val': init_bias[ii]}, wd=wd, name='b_{}'.format(ii + 1))

        # initialize cluster center and label
        c_center = [[] for _ in xrange(num_layer - 1)]
        c_label = [[] for _ in xrange(num_layer - 1)]

        for ii in xrange(num_layer - 1):
            c_center[ii] = weight_variable([num_cluster[ii], c_shape[ii][1]], init_method='truncated_normal', init_param={
                                           'mean': 0.0, 'stddev': init_std[ii]}, name='cluster_center_{}'.format(ii + 1), trainable=False)

            if c_shape[ii][0] < num_cluster[ii]:
                random_init_label = np.random.choice(
                    num_cluster[ii], c_shape[ii][0], replace=False)
            else:
                random_init_label = np.concatenate([np.random.permutation(num_cluster[
                                                   ii]), np.random.choice(num_cluster[ii], c_shape[ii][0] - num_cluster[ii])])

            c_label[ii] = tf.Variable(random_init_label, name='cluster_label_{}'.format(
                ii + 1), trainable=False, dtype=tf.int64)

        # load existed model
        ops['load_weights'] = tf.group(
            w[0].assign(model_w1),
            w[1].assign(model_w2),
            w[2].assign(model_w3),
            w[3].assign(model_w4),
            w[4].assign(model_w5),
            w[5].assign(model_w6),
            w[6].assign(model_w7),
            b[0].assign(model_b1),
            b[1].assign(model_b2),
            b[2].assign(model_b3),
            b[3].assign(model_b4),
            b[4].assign(model_b5),
            b[5].assign(model_b6),
            b[6].assign(model_b7)
        )

        # build computation graph
        # layer 1
        h1 = tf.nn.conv2d(input=input_images, filter=w[0], strides=[
                          1, 4, 4, 1], padding='VALID') + b[0]

        # layer 1 clustering
        e1 = get_conv_cluster_embedding(c_type[0], c_shape[0][1], h1)
        reset_1 = tf.scatter_update(c_center[0], c_reset_idx[
                                    0], tf.gather(e1, s_reset_idx[0]))
        c_update_1 = kmeans_clustering(e1, c_center[0], c_label[
                                       0], num_cluster[0], input_eta)
        reg_1 = compute_reg(e1, c_center[0], c_label[0], c_alpha[0])

        h1 = tf.nn.relu(h1, name='relu1')
        h1 = tf.nn.lrn(h1, depth_radius=2, bias=1.0,
                       alpha=2.0e-5, beta=0.75, name='lrn1')
        h1 = tf.nn.max_pool(h1, ksize=[1, 3, 3, 1], strides=[
                            1, 2, 2, 1], padding='VALID', name='pool1')

        # layer 2, two towers
        h2_l, h2_r = tf.split(split_dim=3, num_split=2, value=h1)
        w2_l, w2_r = tf.split(split_dim=3, num_split=2, value=w[1])
        b2_l, b2_r = tf.split(split_dim=0, num_split=2, value=b[1])

        h2_l = tf.nn.conv2d(input=h2_l, filter=w2_l, strides=[
                            1, 1, 1, 1], padding='SAME') + b2_l
        h2_r = tf.nn.conv2d(input=h2_r, filter=w2_r, strides=[
                            1, 1, 1, 1], padding='SAME') + b2_r
        h2 = tf.concat(concat_dim=3, values=[h2_l, h2_r])

        # layer 2 clustering
        e2 = get_conv_cluster_embedding(c_type[1], c_shape[1][1], h2)
        reset_2 = tf.scatter_update(c_center[1], c_reset_idx[
                                    1], tf.gather(e2, s_reset_idx[1]))
        c_update_2 = kmeans_clustering(e2, c_center[1], c_label[
                                       1], num_cluster[1], input_eta)
        reg_2 = compute_reg(e2, c_center[1], c_label[1], c_alpha[1])

        h2 = tf.nn.relu(h2, name='relu2')
        h2 = tf.nn.local_response_normalization(
            h2, depth_radius=2, bias=1.0, alpha=2.0e-5, beta=0.75, name='lrn2')
        h2 = tf.nn.max_pool(h2, ksize=[1, 3, 3, 1], strides=[
                            1, 2, 2, 1], padding='VALID', name='pool2_right')

        # layer 3
        h3 = tf.nn.conv2d(input=h2, filter=w[2], strides=[
                          1, 1, 1, 1], padding='SAME') + b[2]

        # layer 3 clustering
        e3 = get_conv_cluster_embedding(c_type[2], c_shape[2][1], h3)
        reset_3 = tf.scatter_update(c_center[2], c_reset_idx[
                                    2], tf.gather(e3, s_reset_idx[2]))
        c_update_3 = kmeans_clustering(e3, c_center[2], c_label[
                                       2], num_cluster[2], input_eta)
        reg_3 = compute_reg(e3, c_center[2], c_label[2], c_alpha[2])

        h3 = tf.nn.relu(h3, name='relu3')

        # layer 4, two towers
        h4_l, h4_r = tf.split(split_dim=3, num_split=2, value=h3)
        w4_l, w4_r = tf.split(split_dim=3, num_split=2, value=w[3])
        b4_l, b4_r = tf.split(split_dim=0, num_split=2, value=b[3])

        h4_l = tf.nn.conv2d(input=h4_l, filter=w4_l, strides=[
                            1, 1, 1, 1], padding='SAME') + b4_l
        h4_r = tf.nn.conv2d(input=h4_r, filter=w4_r, strides=[
                            1, 1, 1, 1], padding='SAME') + b4_r

        # layer 4 clustering
        h4 = tf.concat(concat_dim=3, values=[h4_l, h4_r])

        e4 = get_conv_cluster_embedding(c_type[3], c_shape[3][1], h4)
        reset_4 = tf.scatter_update(c_center[3], c_reset_idx[
                                    3], tf.gather(e4, s_reset_idx[3]))
        c_update_4 = kmeans_clustering(e4, c_center[3], c_label[
                                       3], num_cluster[3], input_eta)
        reg_4 = compute_reg(e4, c_center[3], c_label[3], c_alpha[3])

        h4_l = tf.nn.relu(h4_l, name='relu4_left')
        h4_r = tf.nn.relu(h4_r, name='relu4_right')

        # layer 5
        w5_l, w5_r = tf.split(split_dim=3, num_split=2, value=w[4])
        b5_l, b5_r = tf.split(split_dim=0, num_split=2, value=b[4])

        h5_l = tf.nn.conv2d(input=h4_l, filter=w5_l, strides=[
                            1, 1, 1, 1], padding='SAME') + b5_l
        h5_r = tf.nn.conv2d(input=h4_r, filter=w5_r, strides=[
                            1, 1, 1, 1], padding='SAME') + b5_r

        # layer 5 clustering
        h5 = tf.concat(concat_dim=3, values=[h5_l, h5_r])

        e5 = get_conv_cluster_embedding(c_type[4], c_shape[4][1], h5)
        reset_5 = tf.scatter_update(c_center[4], c_reset_idx[
                                    4], tf.gather(e5, s_reset_idx[4]))
        c_update_5 = kmeans_clustering(e5, c_center[4], c_label[
                                       4], num_cluster[4], input_eta)
        reg_5 = compute_reg(e5, c_center[4], c_label[4], c_alpha[4])

        h5 = tf.nn.relu(h5, name='relu5')
        h5 = tf.nn.max_pool(h5, ksize=[1, 3, 3, 1], strides=[
                            1, 2, 2, 1], padding='VALID', name='pool5')

        # layer 6
        if param['using_caffe_weights']:
            h5 = tf.transpose(h5, perm=[0, 3, 2, 1])    # BWHC -> BCHW

        h5 = tf.reshape(h5, shape=[-1, 9216])
        h6 = tf.matmul(h5, w[5]) + b[5]

        # layer 6 clustering
        reset_6 = tf.scatter_update(c_center[5], c_reset_idx[
                                    5], tf.gather(h6, s_reset_idx[5]))
        c_update_6 = kmeans_clustering(h6, c_center[5], c_label[
                                       5], num_cluster[5], input_eta)
        reg_6 = compute_reg(h6, c_center[5], c_label[5], c_alpha[5])

        h6_rep = tf.nn.relu(h6)
        prob6 = 1 + (dropout_rate - 1) * tf.to_float(phase_train)
        h6 = tf.nn.dropout(h6_rep, keep_prob=prob6,
                           noise_shape=None, name='dropout_6')

        # layer 7
        h7 = tf.matmul(h6, w[6]) + b[6]
        h7_rep = tf.matmul(h6_rep, w[6]) + b[6]

        # layer 7 clustering
        reset_7 = tf.scatter_update(c_center[6], c_reset_idx[
                                    6], tf.gather(h7_rep, s_reset_idx[6]))
        c_update_7 = kmeans_clustering(h7_rep, c_center[6], c_label[
                                       6], num_cluster[6], input_eta)
        reg_7 = compute_reg(h7_rep, c_center[6], c_label[6], c_alpha[6])

        h7 = tf.nn.relu(h7)
        prob7 = 1 + (dropout_rate - 1) * tf.to_float(phase_train)
        h7 = tf.nn.dropout(h7, keep_prob=prob7,
                           noise_shape=None, name='dropout_7')

        # layer 8
        logits = tf.matmul(h7, w[7]) + b[7]
        ops['scaled_logits'] = tf.nn.softmax(logits)

        # output ops
        ops['embeddings'] = [e1, e2, e3, e4, e5, h6, h7]
        ops['cluster_center'] = c_center
        ops['cluster_label'] = c_label
        ops['clustering_ops'] = [c_update_1, c_update_2, c_update_3,
                                 c_update_4, c_update_5, c_update_6, c_update_7]
        ops['reg_ops'] = [reg_1, reg_2, reg_3, reg_4, reg_5, reg_6, reg_7]
        ops['reset_ops'] = [reset_1, reset_2,
                            reset_3, reset_4, reset_5, reset_6, reset_7]

        # compute cross-entropy loss
        reg_term = tf.reduce_sum(tf.pack(ops['reg_ops']))
        CE_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, input_labels))
        ops['CE_loss'] = CE_loss

        # setting optimization
        global_step = tf.Variable(0.0, trainable=False)
        learn_rate = tf.train.exponential_decay(param['base_learn_rate'], global_step, param[
                                                'learn_rate_decay_step'], param['learn_rate_decay_rate'], staircase=True)

        ops['train_step'] = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=param[
            'momentum']).minimize(CE_loss + reg_term)

    return ops
