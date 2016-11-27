"""run_zero_shot.py

Usage:
  run_zero_shot.py <exp_id>
"""
import os
import math
import numpy as np
import exp_config as cg
import tensorflow as tf
import cPickle as pickle

from docopt import docopt
from CUB_input import read_CUB
from mini_batch_iter import MiniBatchIterator
from AlexNet import build_AlexNet_CUB
from AlexNet_clustering import build_AlexNet_CUB_clustering


def learn_zero_shot(X_train, X_val, Y_train, Y_val, phi, mask_idx, param):
    """ Learn parameters for zero-shot learning
        SGD for unregularized Structured SVM

        Input:
            X: N X D input feature 
            Y: N X C class label
            phi: C X M output embedding
            param: hyper parameters

        Output:
            W: D X M
    """

    num_sample = X_train.shape[0]
    dim_in = X_train.shape[1]
    num_class = Y_val.shape[1]
    dim_out = phi.shape[1]
    W = np.random.rand(dim_in, dim_out) * 0.1     # D X_train M
    init_lr = param['eta']

    # SGD
    train_iterator = MiniBatchIterator(idx_start=0, bat_size=param['batch_size'], num_sample=param[
                                       'num_train_imgs'], train_phase=True, is_permute=True)

    for ii in xrange(param['num_train_iter']):
        idx_bat = train_iterator.get_batch()
        x = X_train[idx_bat]
        y = Y_train[idx_bat]
        y_idx = np.argmax(y, axis=1)

        # search
        y_pred, score = compute_argmax(x, y, W, phi)
        loss = np.amax(score, axis=1)
        loss[loss < 0] = 0

        print 'Iter = {:07d} || Loss = {:e}'.format(ii + 1, np.mean(loss))

        # evaluate gradient
        dW = np.zeros([dim_in, dim_out])
        for jj in xrange(y_pred.shape[0]):
            tmp_x = np.expand_dims(x[jj, :], axis=1)
            tmp_y = np.expand_dims(
                phi[y_idx[jj], :] - phi[y_pred[jj], :], axis=1).T

            dW += np.dot(tmp_x, tmp_y)

        W += init_lr * dW / float(param['batch_size'])

        if (ii + 1) % param['lr_decay_iter'] == 0:
            init_lr *= param['lr_decay_rate']

        if (ii + 1) % param['val_iter'] == 0 or ii == 0:
            Y_pred, score = predict_zero_shot(X_val, W, phi, mask_idx)

            acc = np.sum(np.array(Y_pred == np.argmax(Y_val, axis=1)
                                  ).astype(np.float)) / Y_val.shape[0]
            print 'Valid acc @iter{:06d} = {:5.2f}'.format(ii + 1, acc * 100)

    return W


def compute_argmax(X, Y, W, phi):
    """ Inference """

    N = X.shape[0]
    theta = np.dot(X, W)  # N X M
    delta = 1 - Y   # N X C
    alpha = np.dot(theta, phi.T)    # N X C
    Y_idx = np.argmax(Y, axis=1)

    #
    score = delta + alpha - \
        np.expand_dims(alpha[np.array(range(N)), Y_idx], axis=1)

    Y_pred = np.argmax(score, axis=1)

    return Y_pred, score


def predict_zero_shot(X, W, phi, mask_idx):
    """ Inference for zero-shot """

    N = X.shape[0]
    C = phi.shape[0]
    theta = np.dot(X, W)  # N X M

    #
    score = np.reshape(
        np.sum(np.tile(theta, [C, 1]) * np.repeat(phi, N, axis=0), axis=1), [C, N]).T
    score[:, mask_idx] = -np.inf
    Y_pred = np.argmax(score, axis=1)

    return Y_pred, score


def zero_shot_split(data, random_split=True, train_class=None, val_class=None, test_class=None):

    X_all = np.concatenate([data['train_feat'], data['test_feat']], axis=0)
    Y_all_1_hot = np.concatenate(
        [data['train_label'], data['test_label']], axis=0)
    Y_all = np.argmax(Y_all_1_hot, axis=1)

    if random_split:
        unique_label = np.unique(Y_all)
        rand_label = np.random.permutation(unique_label)
        seen_class = rand_label[:150]
        unseen_class = rand_label[150:]

    idx_test = []
    for ii in test_class:
        idx_test += [np.nonzero(Y_all == ii)[0]]

    idx_val = []
    for ii in val_class:
        idx_val += [np.nonzero(Y_all == ii)[0]]

    idx_test = np.concatenate(idx_test)
    idx_val = np.concatenate(idx_val)
    idx_train = np.setdiff1d(
        np.arange(X_all.shape[0]), np.concatenate([idx_test, idx_val]))

    X_train = X_all[idx_train]
    Y_train = Y_all_1_hot[idx_train]
    X_val = X_all[idx_val]
    Y_val = Y_all_1_hot[idx_val]
    X_test = X_all[idx_test]
    Y_test = Y_all_1_hot[idx_test]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def extract_feat(feat_file, param):
    # read data from file
    print 'Reading data start!'
    input_data = read_CUB(param['train_list_file'], param['test_list_file'])
    input_data['mean_img'] = pickle.load(open(param['mean_img'], 'rb'))
    input_data['train_img'] = np.concatenate(
        [input_data['train_img'], input_data['val_img']], axis=0)
    input_data['train_label'] = np.concatenate(
        [input_data['train_label'], input_data['val_label']])
    print 'Reading data done!'

    # build model
    model_ops = build_AlexNet_CUB(param)
    print 'Building model done!'

    # tf set up
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    saver.restore(sess, os.path.join(
        param['test_folder'], param['test_model_name']))

    num_train_img = input_data['train_img'].shape[0]
    num_test_img = input_data['test_img'].shape[0]
    train_feat = []
    test_feat = []
    train_label = np.zeros([num_train_img, param['label_size']])
    test_label = np.zeros([num_test_img, param['label_size']])

    feed_data = {
        model_ops['phase_train']: False,
        model_ops['dropout_rate']: param['dropout_rate']

    }

    num_test_iter = int(math.ceil(float(num_train_img) / param['bat_size']))
    test_iterator = MiniBatchIterator(idx_start=0, bat_size=param[
                                      'bat_size'], num_sample=num_train_img, train_phase=False, is_permute=False)
    for test_iter in xrange(num_test_iter):
        # generate a batch
        idx_bat = test_iterator.get_batch()
        bat_imgs = input_data['train_img'][idx_bat, :, :,
                                           :].astype('float32') - input_data['mean_img']
        bat_labels = input_data['train_label'][idx_bat].astype('int32')
        feed_data[model_ops['input_images']] = bat_imgs

        # run a batch
        ee = sess.run(model_ops['embeddings'], feed_dict=feed_data)
        train_feat += [ee[6]]
        train_label[idx_bat, bat_labels] = 1

        print '{:06d}-th batch'.format(test_iter)

    num_test_iter = int(math.ceil(float(num_test_img) / param['bat_size']))
    test_iterator = MiniBatchIterator(idx_start=0, bat_size=param[
                                      'bat_size'], num_sample=num_test_img, train_phase=False, is_permute=False)

    for test_iter in xrange(num_test_iter):
        # generate a batch
        idx_bat = test_iterator.get_batch()
        bat_imgs = input_data['test_img'][idx_bat, :, :, :].astype(
            'float32') - input_data['mean_img']
        bat_labels = input_data['test_label'][idx_bat].astype('int32')
        feed_data[model_ops['input_images']] = bat_imgs

        # run a batch
        ee = sess.run(model_ops['embeddings'], feed_dict=feed_data)
        test_feat += [ee[6]]
        test_label[idx_bat, bat_labels] = 1

        print '{:06d}-th batch'.format(test_iter)

    # save feat to disk
    np.savez(feat_file, train_feat=train_feat, test_feat=test_feat,
             train_label=train_label, test_label=test_label)

    sess.close()

    data = {
        'train_feat': np.concatenate(train_feat, axis=0),
        'train_label': train_label,
        'test_feat': np.concatenate(test_feat, axis=0),
        'test_label': test_label
    }

    return data


def main():
    # get exp parameters
    args = docopt(__doc__)
    param = getattr(cg, args['<exp_id>'])()

    # read feat
    if param['load_feat'] == True:
        data = np.load(param['feat_file'])
    else:
        data = extract_feat(param['feat_file'], param['model_config'])

    # read embedding matrix
    phi = np.loadtxt(param['embedding_file'])
    phi = phi.astype(np.float32) / 100.0
    phi = phi / np.expand_dims(np.sqrt(np.sum(phi * phi, axis=1)), axis=1)

    # read split
    split_vars = np.load(param['split_file'])
    class_idx_train = split_vars['class_idx_train']
    class_idx_val = split_vars['class_idx_val']
    class_idx_test = split_vars['class_idx_test']

    # select 150 as seen and 50 as unseen data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = zero_shot_split(
        data, random_split=False, train_class=class_idx_train, val_class=class_idx_val, test_class=class_idx_test)

    # learn the zero shot parameter
    W = learn_zero_shot(X_train, X_val, Y_train, Y_val, phi, np.concatenate(
        [class_idx_train, class_idx_test]), param['ssvm_param'])

    # test
    if param['run_test'] == True:
        Y_pred, score = predict_zero_shot(
            X_test, W, phi, np.concatenate([class_idx_train, class_idx_val]))
        acc = np.sum(np.array(Y_pred == np.argmax(Y_test, axis=1)
                              ).astype(np.float)) / Y_test.shape[0]

        print 'Zero shot test acc = {:5.2f}'.format(acc * 100)

if __name__ == '__main__':
    main()
