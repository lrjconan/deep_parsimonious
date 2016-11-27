"""eval_clustering.py

Usage:
  eval_clustering.py <exp_id>
"""
import os
import time
import math
import numpy as np
import tensorflow as tf
import nn_cell_lib as nn
import exp_config as cg

from docopt import docopt
from mini_batch_iter import MiniBatchIterator
from CIFAR_input import read_CIFAR10, read_CIFAR100
from CIFAR_models import baseline_model, clustering_model
from scipy.cluster.vq import kmeans2
from scipy.spatial import distance

EPS = 1.0e-16


def get_num(x):
    return len(np.nonzero(x)[0])


def compute_mutual_information(cluster_label, class_label, num_cluster, num_class):

    MI = 0.0
    N = float(len(cluster_label))

    for ii in xrange(num_class):
        for jj in xrange(num_cluster):
            idx_cluster = cluster_label == jj
            idx_class = class_label == ii
            W = get_num(idx_cluster)
            C = get_num(idx_class)
            J = get_num(idx_cluster & idx_class)
            MI += J / N * np.log((N * J + EPS) / (W * C + EPS))

    return MI


def compute_entropy(class_label, num_class):
    H = 0.0
    N = float(len(class_label))

    for ii in xrange(num_class):
        idx_class = class_label == ii
        C = get_num(idx_class)
        H -= C / N * np.log((C + EPS) / N)

    return H


def compute_normalized_mutual_information(cluster_label, class_label, num_cluster, num_class):
    MI = compute_mutual_information(
        cluster_label, class_label, num_cluster, num_class)

    H1 = compute_entropy(class_label, num_class)
    H2 = compute_entropy(cluster_label, num_cluster)

    NMI = 2 * MI / (H1 + H2)

    return NMI


def Kmeans(data, num_K):
    centroid, label = kmeans2(
        data=data, k=num_K, iter=100, minit='points', missing='warn')
    return centroid, label


def main():
    # get exp parameters
    args = docopt(__doc__)
    param = getattr(cg, args['<exp_id>'])()

    # read data from file
    if param['dataset_name'] == 'CIFAR10':
        input_data = read_CIFAR10(param['data_folder'])
    elif param['dataset_name'] == 'CIFAR100':
        input_data = read_CIFAR100(param['data_folder'])
    else:
        raise ValueError('Unsupported dataset name!')
    print 'Reading data done!'

    # build model
    test_op_names = ['embeddings']

    if param['model_name'] == 'baseline':
        model_ops = baseline_model(param)
    elif param['model_name'] == 'parsimonious':
        model_ops = clustering_model(param)
    else:
        raise ValueError('Unsupported model name!')

    test_ops = [model_ops[i] for i in test_op_names]
    print 'Building model done!'

    # run model
    input_data['train_img'] = np.concatenate(
        [input_data['train_img'], input_data['val_img']], axis=0)
    input_data['train_label'] = np.concatenate(
        [input_data['train_label'], input_data['val_label']])

    num_train_img = input_data['train_img'].shape[0]
    max_test_iter = int(math.ceil(num_train_img / param['bat_size']))
    test_iterator = MiniBatchIterator(idx_start=0, bat_size=param[
                                      'bat_size'], num_sample=num_train_img, train_phase=False, is_permute=False)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(
        param['test_folder'], param['test_model_name']))
    print 'Graph initialization done!'

    if param['model_name'] == 'parsimonious':
        param['num_layer_cnn'] = len(
            [xx for xx in param['num_cluster_cnn'] if xx])
        param['num_layer_mlp'] = len(
            [xx for xx in param['num_cluster_mlp'] if xx])
        num_layer_reg = param['num_layer_cnn'] + param['num_layer_mlp']

        cluster_center = sess.run(model_ops['cluster_center'])

    embeddings = [[] for _ in xrange(num_layer_reg)]

    for test_iter in xrange(max_test_iter):
        idx_bat = test_iterator.get_batch()

        bat_imgs = (input_data['train_img'][idx_bat, :, :, :].astype(
            np.float32) - input_data['mean_img']) / 255.0

        feed_data = {model_ops['input_images']: bat_imgs}

        results = sess.run(test_ops, feed_dict=feed_data)

        test_results = {}
        for res, name in zip(results, test_op_names):
            test_results[name] = res

        for ii, ee in enumerate(test_results['embeddings']):
            if ii < 3:
                continue

            embeddings[ii] += [ee]

    for ii in xrange(num_layer_reg):
        if ii < 3:
            continue

        embeddings[ii] = np.concatenate(embeddings[ii], axis=0)

        # kmeans
        centroid, tmp_label = Kmeans(embeddings[ii], 100)
        cluster_center[ii] = centroid

        # deep clustering
        pdist = distance.cdist(
            cluster_center[ii], embeddings[ii], 'sqeuclidean')

        tmp_label = np.argsort(pdist, axis=0)[0]
        sort_idx = np.argsort(pdist, axis=1)

        NMI = compute_normalized_mutual_information(tmp_label, input_data['train_label'].astype(
            np.int), cluster_center[ii].shape[0], param['label_size'])
        print 'NMI = {}'.format(NMI)

    sess.close()

if __name__ == '__main__':
    main()
