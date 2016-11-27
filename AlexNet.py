import numpy as np
import tensorflow as tf
from nn_cell_lib import weight_variable


def load_caffe_model(caffe_model_file, model_ops):
    num_layer = 8
    w = [[] for _ in xrange(num_layer)]
    b = [[] for _ in xrange(num_layer)]
    weight = np.fromfile(caffe_model_file, dtype=np.float32)

    count = 0
    size_w = [11 * 11 * 3 * 96, 5 * 5 * 48 * 256, 3 * 3 * 256 * 384,
              3 * 3 * 192 * 384, 3 * 3 * 192 * 256, 9216 * 4096, 4096 * 4096,
              4096 * 1000]
    shape_w = [[11, 11, 3, 96], [5, 5, 48, 256], [3, 3, 256, 384],
               [3, 3, 192, 384], [3, 3, 192, 256], [9216, 4096], [4096, 4096],
               [4096, 1000]]

    for ii in xrange(num_layer - 1, -1, -1):
        C_shape = shape_w[ii]
        F_shape = list(reversed(C_shape))
        w_ = np.reshape(weight[count: count + size_w[ii]], F_shape)
        w[ii] = np.transpose(w_)
        # print 'w', ii, w[ii].shape
        count += size_w[ii]

    for ii in xrange(num_layer):
        b[ii] = np.reshape(
            weight[count: count + shape_w[ii][-1]], shape_w[ii][-1])
        # print 'b', ii, b[ii].shape
        count += shape_w[ii][-1]

    feed_weight = {
        model_ops['model_w1']: w[0],
        model_ops['model_w2']: w[1],
        model_ops['model_w3']: w[2],
        model_ops['model_w4']: w[3],
        model_ops['model_w5']: w[4],
        model_ops['model_w6']: w[5],
        model_ops['model_w7']: w[6],
        model_ops['model_b1']: b[0],
        model_ops['model_b2']: b[1],
        model_ops['model_b3']: b[2],
        model_ops['model_b4']: b[3],
        model_ops['model_b5']: b[4],
        model_ops['model_b6']: b[5],
        model_ops['model_b7']: b[6]
    }

    return feed_weight


def build_AlexNet_CUB(param):
    """ Build AlexNet for CUB dataset 
        Note: if use caffe-compatible weights, the input should be BWHC
    """
    ops = {}
    num_layer = 8
    wd = None if param['weight_decay'] == 0 else param['weight_decay']
    f_shape = param['filter_shape']
    init_std = param['init_std']
    init_bias = param['init_bias']

    with tf.device(param['device']):
        input_images = tf.placeholder(tf.float32, [None, 227, 227, 3])
        input_labels = tf.placeholder(tf.int32, [None])
        learn_rate = tf.placeholder(tf.float32, [])
        dropout_rate = tf.placeholder(tf.float32, [])
        phase_train = tf.placeholder(tf.bool, [])

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

        ops['input_images'] = input_images
        ops['input_labels'] = input_labels
        ops['learn_rate'] = learn_rate
        ops['phase_train'] = phase_train
        ops['dropout_rate'] = dropout_rate

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

        # load existed model
        ops['load_weights'] = tf.group(
            w[0].assign(model_w1),
            w[1].assign(model_w2),
            w[2].assign(model_w3),
            w[3].assign(model_w4),
            w[4].assign(model_w5),
            w[5].assign(model_w6),
            w[6].assign(model_w7),
            # w[7].assign(model_w8),
            b[0].assign(model_b1),
            b[1].assign(model_b2),
            b[2].assign(model_b3),
            b[3].assign(model_b4),
            b[4].assign(model_b5),
            b[5].assign(model_b6),
            b[6].assign(model_b7)
            # b[7].assign(model_b8)
        )

        # build computation graph
        # layer 1
        h1 = tf.nn.conv2d(input=input_images, filter=w[0], strides=[
                          1, 4, 4, 1], padding='VALID') + b[0]
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

        h2 = tf.nn.relu(h2, name='relu2')
        h2 = tf.nn.local_response_normalization(
            h2, depth_radius=2, bias=1.0, alpha=2.0e-5, beta=0.75, name='lrn2')
        h2 = tf.nn.max_pool(h2, ksize=[1, 3, 3, 1], strides=[
                            1, 2, 2, 1], padding='VALID', name='pool2_right')

        # layer 3
        h3 = tf.nn.conv2d(input=h2, filter=w[2], strides=[
                          1, 1, 1, 1], padding='SAME') + b[2]
        h3 = tf.nn.relu(h3, name='relu3')

        # layer 4, two towers
        h4_l, h4_r = tf.split(split_dim=3, num_split=2, value=h3)
        w4_l, w4_r = tf.split(split_dim=3, num_split=2, value=w[3])
        b4_l, b4_r = tf.split(split_dim=0, num_split=2, value=b[3])

        h4_l = tf.nn.conv2d(input=h4_l, filter=w4_l, strides=[
                            1, 1, 1, 1], padding='SAME') + b4_l
        h4_l = tf.nn.relu(h4_l, name='relu4_left')

        h4_r = tf.nn.conv2d(input=h4_r, filter=w4_r, strides=[
                            1, 1, 1, 1], padding='SAME') + b4_r
        h4_r = tf.nn.relu(h4_r, name='relu4_right')

        # layer 5
        w5_l, w5_r = tf.split(split_dim=3, num_split=2, value=w[4])
        b5_l, b5_r = tf.split(split_dim=0, num_split=2, value=b[4])

        h5_l = tf.nn.conv2d(input=h4_l, filter=w5_l, strides=[
                            1, 1, 1, 1], padding='SAME') + b5_l
        h5_l = tf.nn.relu(h5_l, name='relu5_left')
        h5_r = tf.nn.conv2d(input=h4_r, filter=w5_r, strides=[
                            1, 1, 1, 1], padding='SAME') + b5_r
        h5_r = tf.nn.relu(h5_r, name='relu5_right')
        h5 = tf.concat(concat_dim=3, values=[h5_l, h5_r])
        h5 = tf.nn.max_pool(h5, ksize=[1, 3, 3, 1], strides=[
                            1, 2, 2, 1], padding='VALID', name='pool5')

        # layer 6
        if param['using_caffe_weights']:
            h5 = tf.transpose(h5, perm=[0, 3, 2, 1])    # BWHC -> BCHW

        h5 = tf.reshape(h5, shape=[-1, 9216])
        h6 = tf.nn.relu(tf.matmul(h5, w[5]) + b[5])

        prob6 = 1 + (dropout_rate - 1) * tf.to_float(phase_train)
        h6 = tf.nn.dropout(h6, keep_prob=prob6,
                           noise_shape=None, name='dropout_6')

        # layer 7
        h7 = tf.nn.relu(tf.matmul(h6, w[6]) + b[6])
        prob7 = 1 + (dropout_rate - 1) * tf.to_float(phase_train)
        h7 = tf.nn.dropout(h7, keep_prob=prob7,
                           noise_shape=None, name='dropout_7')

        # layer 8
        logits = tf.matmul(h7, w[7]) + b[7]
        ops['scaled_logits'] = tf.nn.softmax(logits)
        ops['embeddings'] = [h1, h2, h3, h4_l, h5, h6, h7]

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
