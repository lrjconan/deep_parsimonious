"""run_train_model.py

Usage:
  run_train_model.py <exp_id>
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
from kmeans_plus_plus import get_update_cluster_idx
from CUB_input import read_CUB
from CIFAR_input import read_CIFAR10, read_CIFAR100
from CIFAR_models import baseline_model, clustering_model
from AlexNet import load_caffe_model, build_AlexNet_CUB
from AlexNet_clustering import build_AlexNet_CUB_clustering


def update_cluster_centers(sess, input_data, model_ops, hist_label, train_iterator, param, hist_thresh=10):
    """ Update Cluster Centers """
    num_cluster = param['num_cluster']
    is_update = [False if xx is not None else True for xx in num_cluster]
    mask_update = [np.zeros(xx, dtype=np.bool)
                   if xx is not None else None for xx in num_cluster]

    for ll, hh in enumerate(hist_label):
        if hh is not None:
            for kk in xrange(hh.shape[0]):
                if hh[kk] >= hist_thresh:
                    mask_update[ll][kk] = True
                else:
                    is_update[ll] = True

    if all(is_update):
        num_pass = 0
        for ii, xx in enumerate(mask_update):
            if xx is not None:
                num_empty = num_cluster[ii] - len(np.nonzero(xx))

                if num_empty > num_pass:
                    num_pass = num_empty

        idx_start = train_iterator.idx_start
        num_pass = int(math.ceil(num_pass / float(param['bat_size'])))
        c_center = sess.run(model_ops['cluster_center'])

        for ii in xrange(num_pass):
            # generate a mini-batch
            idx_train_bat = train_iterator.get_batch()

            bat_imgs = (input_data['train_img'][idx_train_bat, :, :, :].astype(
                np.float32) - input_data['mean_img']) / param['denom_const']

            feed_data = {model_ops['input_images']: bat_imgs,
                         model_ops['input_eta']: param['eta']}

            if param['dataset_name'] == 'CUB':
                feed_data[model_ops['dropout_rate']] = param['dropout_rate']
                feed_data[model_ops['phase_train']] = True

            embeddings = sess.run(model_ops['embeddings'], feed_dict=feed_data)

            # generate new cluster center
            idx_center, idx_sample = get_update_cluster_idx(
                c_center, embeddings, mask_update)

            var_keys = [model_ops['input_images']]
            var_names = [feed_data[model_ops['input_images']]]

            if param['dataset_name'] == 'CIFAR10' or param['dataset_name'] == 'CIFAR100':
                num_layer_cnn = len(param['num_cluster_cnn'])
                num_layer_mlp = len(param['num_cluster_mlp'])

                var_keys += [model_ops['c_reset_idx_cnn'][ii]
                             for ii in xrange(num_layer_cnn)]
                var_keys += [model_ops['s_reset_idx_cnn'][ii]
                             for ii in xrange(num_layer_cnn)]
                var_keys += [model_ops['c_reset_idx_mlp'][ii]
                             for ii in xrange(num_layer_mlp)]
                var_keys += [model_ops['s_reset_idx_mlp'][ii]
                             for ii in xrange(num_layer_mlp)]

                var_names += idx_center[:num_layer_cnn]
                var_names += idx_sample[:num_layer_cnn]
                var_names += idx_center[num_layer_cnn:]
                var_names += idx_sample[num_layer_cnn:]
            elif param['dataset_name'] == 'CUB':
                num_layer_reg = param['num_layer_reg']

                var_keys += [model_ops['dropout_rate'],
                             model_ops['phase_train']]
                var_keys += [model_ops['c_reset_idx'][ii]
                             for ii in xrange(num_layer_reg)]
                var_keys += [model_ops['s_reset_idx'][ii]
                             for ii in xrange(num_layer_reg)]

                var_names += [param['dropout_rate'], True]
                var_names += idx_center[:num_layer_reg]
                var_names += idx_sample[:num_layer_reg]
            else:
                raise ValueError('Unsupported dataset name!')

            feed_data_reset = dict(zip(var_keys, var_names))
            sess.run(model_ops['reset_ops'], feed_dict=feed_data_reset)

        # reset iterator
        train_iterator.reset_iterator(idx_start)


def main():
    # get exp parameters
    args = docopt(__doc__)
    param = getattr(cg, args['<exp_id>'])()

    if param['resume_training'] == True:
        param['exp_id'] = param['resume_exp_id']
    else:
        param['exp_id'] = args['<exp_id>'] + '_' + \
            time.strftime("%Y-%b-%d-%H-%M-%S")

    param['save_folder'] = os.path.join(param['save_path'], param['exp_id'])

    # save parameters
    if not os.path.isdir(param['save_folder']):
        os.mkdir(param['save_folder'])

    with open(os.path.join(param['save_folder'], 'hyper_param.txt'), 'w') as f:
        for key, value in param.iteritems():
            f.write('{}: {}\n'.format(key, value))

    if param['model_name'] == 'parsimonious':
        if param['dataset_name'] == 'CIFAR10' or param['dataset_name'] == 'CIFAR100':
            param['num_layer_cnn'] = len(
                [xx for xx in param['num_cluster_cnn'] if xx])
            param['num_layer_mlp'] = len(
                [xx for xx in param['num_cluster_mlp'] if xx])
            param['num_cluster'] = param[
                'num_cluster_cnn'] + param['num_cluster_mlp']
            num_layer_reg = param['num_layer_cnn'] + param['num_layer_mlp']
        elif param['dataset_name'] == 'CUB':
            num_layer_reg = len(np.nonzero(np.array(param['num_cluster']))[0])
        else:
            raise ValueError('Unsupported dataset name!')

        param['num_layer_reg'] = num_layer_reg
        hist_label = [np.zeros(xx) if xx is not None else None for xx in param[
            'num_cluster']]
        reg_val = np.zeros(num_layer_reg)

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

    train_op_names = ['train_step', 'CE_loss']
    val_op_names = ['scaled_logits']
    train_ops = [model_ops[i] for i in train_op_names]
    val_ops = [model_ops[i] for i in val_op_names]
    print 'Building model done!'

    # run model
    if param['merge_valid']:
        input_data['train_img'] = np.concatenate(
            [input_data['train_img'], input_data['val_img']], axis=0)
        input_data['train_label'] = np.concatenate(
            [input_data['train_label'], input_data['val_label']])

    num_train_img = input_data['train_img'].shape[0]
    num_val_img = input_data['test_img'].shape[0]
    epoch_iter = int(math.ceil(num_train_img / param['bat_size']))
    max_val_iter = int(math.ceil(num_val_img / param['bat_size']))
    train_iterator = MiniBatchIterator(idx_start=0, bat_size=param[
                                       'bat_size'], num_sample=num_train_img, train_phase=True, is_permute=True)
    val_iterator = MiniBatchIterator(idx_start=0, bat_size=param[
                                     'bat_size'], num_sample=num_val_img, train_phase=False, is_permute=False)

    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    train_iter_start = 0
    if param['resume_training'] == True:
        saver.restore(sess, os.path.join(
            param['save_folder'], param['resume_model_name']))
        train_iter_start = param['resume_step']
    else:
        sess.run(tf.initialize_all_variables())
        if param['dataset_name'] == 'CUB' and param['using_caffe_weights'] == True:
            caffe_weight = load_caffe_model(
                param['caffe_model_file'], model_ops)
            sess.run(model_ops['load_weights'], feed_dict=caffe_weight)

    print 'Graph initialization done!'

    for train_iter in xrange(train_iter_start, param['max_train_iter']):
        # generate a batch
        idx_train_bat = train_iterator.get_batch()

        bat_imgs = (input_data['train_img'][idx_train_bat, :, :, :].astype(
            np.float32) - input_data['mean_img']) / param['denom_const']
        bat_labels = input_data['train_label'][idx_train_bat].astype(np.int32)

        feed_data = {
            model_ops['input_images']: bat_imgs,
            model_ops['input_labels']: bat_labels
        }

        if param['dataset_name'] == 'CUB':
            feed_data[model_ops['phase_train']] = True
            feed_data[model_ops['dropout_rate']] = param['dropout_rate']

        # run a batch
        if param['model_name'] == 'baseline':
            results = sess.run(train_ops, feed_dict=feed_data)

            train_results = {}
            for res, name in zip(results, train_op_names):
                train_results[name] = res

            CE_loss = train_results['CE_loss']

        elif param['model_name'] == 'parsimonious':
            feed_data[model_ops['input_eta']] = param['eta']

            # deal with drifted clusters
            if (train_iter + 1) % epoch_iter == 0:
                update_cluster_centers(
                    sess, input_data, model_ops, hist_label, train_iterator, param)

            # get CE/Reg values
            results = sess.run([model_ops['CE_loss']] + model_ops['reg_ops'] +
                               model_ops['cluster_label'], feed_dict=feed_data)
            CE_loss = results[0]
            for ii in xrange(num_layer_reg):
                reg_val[ii] = results[1 + ii]

            cluster_label = results[1 + num_layer_reg:]

            cluster_idx = 0
            for ii, xx in enumerate(param['num_cluster']):
                if xx:
                    tmp_label = cluster_label[cluster_idx]

                    for jj in xrange(tmp_label.shape[0]):
                        hist_label[ii][tmp_label[jj]] += 1

                    cluster_idx += 1

            # run clustering
            if (train_iter + 1) % 1 == 0:
                for iter_clustering in xrange(param['clustering_iter']):
                    sess.run(model_ops['clustering_ops'], feed_dict=feed_data)

            if (train_iter + 1) % epoch_iter == 0:
                for ii in xrange(len(hist_label)):
                    if hist_label[ii] is not None:
                        hist_label[ii].fill(0)

            # run optimization
            sess.run(model_ops['train_step'], feed_dict=feed_data)

        # display statistic
        if (train_iter + 1) % param['disp_iter'] == 0 or train_iter == 0:
            disp_str = 'Train Step = {:06d} || CE loss = {:e}'.format(
                train_iter + 1, CE_loss)

            if param['model_name'] == 'parsimonious':
                disp_str += ' || Clustering '
                for ii in xrange(num_layer_reg):
                    disp_str += 'Reg_{:d} = {:e} '.format(ii + 1, reg_val[ii])

            print disp_str

        # valid model
        if (train_iter + 1) % param['valid_iter'] == 0 or train_iter == 0:
            num_correct = 0.0

            if param['resume_training'] == True:
                print 'Resume Exp ID = {}'.format(param['exp_id'])
            else:
                print 'Exp ID = {}'.format(param['exp_id'])

            for val_iter in xrange(max_val_iter):
                idx_val_bat = val_iterator.get_batch()

                bat_imgs = (input_data['test_img'][idx_val_bat, :, :, :].astype(
                    np.float32) - input_data['mean_img']) / param['denom_const']
                bat_labels = input_data['test_label'][
                    idx_val_bat].astype(np.int32)

                feed_data[model_ops['input_images']] = bat_imgs
                feed_data[model_ops['input_labels']] = bat_labels

                if param['dataset_name'] == 'CUB':
                    feed_data[model_ops['phase_train']] = False

                results = sess.run(val_ops, feed_dict=feed_data)

                val_results = {}
                for res, name in zip(results, val_op_names):
                    val_results[name] = res

                pred_label = np.argmax(val_results['scaled_logits'], axis=1)
                num_correct += np.sum(np.equal(pred_label,
                                               bat_labels).astype(np.float32))

            val_acc = (num_correct / num_val_img)
            print "Val accuracy = {:3f}".format(val_acc * 100)

        # snapshot a model
        if (train_iter + 1) % param['save_iter'] == 0:
            saver.save(sess, os.path.join(param['save_folder'], '{}_snapshot_{:07d}.ckpt'.format(
                param['model_name'], train_iter + 1)))

    sess.close()

if __name__ == '__main__':
    main()
