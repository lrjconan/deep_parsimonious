"""run_test_model.py

Usage:
  run_test_model.py <exp_id>
"""
import os
import time
import math
import numpy as np
import tensorflow as tf
import exp_config as cg
import nn_cell_lib as nn
import cPickle as pickle

from docopt import docopt
from mini_batch_iter import MiniBatchIterator
from CUB_input import read_CUB
from CIFAR_input import read_CIFAR10, read_CIFAR100
from CIFAR_models import baseline_model, clustering_model
from AlexNet import build_AlexNet_CUB
from AlexNet_clustering import build_AlexNet_CUB_clustering


def main():
    # get exp parameters
    args = docopt(__doc__)
    param = getattr(cg, args['<exp_id>'])()

    # read data from file
    param['denom_const'] = 255.0
    if param['dataset_name'] == 'CIFAR10':
        input_data = read_CIFAR10(param['data_folder'])
    elif param['dataset_name'] == 'CIFAR100':
        input_data = read_CIFAR100(param['data_folder'])
    elif param['dataset_name'] == 'CUB':
        param['denom_const'] = 1.0
        input_data = read_CUB(param['train_list_file'], param['test_list_file'])
        input_data['mean_img'] = pickle.load(open(param['mean_img'], 'rb'))
    else:
        raise ValueError('Unsupported dataset name!')
    print 'Reading data done!'

    # build model
    test_op_names = ['scaled_logits']

    # build model
    if param['dataset_name'] == 'CIFAR10' or param['dataset_name'] == 'CIFAR100':
        if param['model_name'] == 'baseline':
            model_ops = baseline_model(param)
        elif param['model_name'] == 'parsimonious':
            model_ops = clustering_model(param)
        else:
            raise ValueError('Unsupported model name!')
    elif param['dataset_name'] == 'CUB':
        if param['model_name'] == 'baseline':
            model_ops = build_AlexNet_CUB(param)
        elif param['model_name'] == 'parsimonious':
            model_ops = build_AlexNet_CUB_clustering(param)
        else:
            raise ValueError('Unsupported model name!')
    else:
        raise ValueError('Unsupported dataset name!')

    test_ops = [model_ops[i] for i in test_op_names]
    print 'Building model done!'

    # run model
    num_test_img = input_data['test_img'].shape[0]
    max_test_iter = int(math.ceil(num_test_img / param['bat_size']))
    test_iterator = MiniBatchIterator(idx_start=0, bat_size=param[
                                      'bat_size'], num_sample=num_test_img, train_phase=False, is_permute=False)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(
        param['test_folder'], param['test_model_name']))
    print 'Graph initialization done!'

    num_correct = 0.0
    for val_iter in xrange(max_test_iter):
        idx_bat = test_iterator.get_batch()

        bat_imgs = (input_data['test_img'][idx_bat, :, :, :].astype(
            np.float32) - input_data['mean_img']) / param['denom_const']
        bat_labels = input_data['test_label'][idx_bat].astype(np.int32)

        feed_data = {
            model_ops['input_images']: bat_imgs,
            model_ops['input_labels']: bat_labels
        }

        if param['dataset_name'] == 'CUB':
            feed_data[model_ops['dropout_rate']] = param['dropout_rate']
            feed_data[model_ops['phase_train']] = False

        results = sess.run(test_ops, feed_dict=feed_data)

        test_results = {}
        for res, name in zip(results, test_op_names):
            test_results[name] = res

        pred_label = np.argmax(test_results['scaled_logits'], axis=1)
        num_correct += np.sum(np.equal(pred_label, bat_labels).astype(float))

    test_acc = (num_correct / num_test_img)
    print 'Test accuracy = {:.3f}'.format(test_acc * 100)

    sess.close()

if __name__ == '__main__':
    main()
