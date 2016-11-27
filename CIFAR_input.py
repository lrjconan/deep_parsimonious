import os
import cPickle
import numpy as np


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def read_CIFAR10(data_folder):
    """ Reads and parses examples from CIFAR10 data files """

    # Constants describing the CIFAR-10 data set.
    img_height = 32
    img_width = 32
    num_class = 10
    num_channel = 3
    num_val_img = 5000

    train_img = []
    train_label = []
    test_img = []
    test_label = []

    train_file_list = ['data_batch_1', 'data_batch_2',
                       'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_file_list = ['test_batch']

    for i in xrange(len(train_file_list)):
        tmp_dict = unpickle(os.path.join(data_folder, train_file_list[i]))
        train_img.append(tmp_dict['data'])
        train_label.append(tmp_dict['labels'])

    tmp_dict = unpickle(os.path.join(data_folder, test_file_list[0]))
    test_img.append(tmp_dict['data'])
    test_label.append(tmp_dict['labels'])

    train_img = np.concatenate(train_img)
    train_label = np.concatenate(train_label)
    test_img = np.concatenate(test_img)
    test_label = np.concatenate(test_label)

    train_img = np.reshape(train_img, [-1, num_channel, img_height, img_width])
    test_img = np.reshape(test_img, [-1, num_channel, img_height, img_width])

    # change format from [B, C, H, W] to [B, H, W, C] for feeding to Tensorflow
    train_img = np.transpose(train_img, [0, 2, 3, 1])
    test_img = np.transpose(test_img, [0, 2, 3, 1])

    mean_img = np.mean(np.concatenate([train_img, test_img]), axis=0)

    # random split for train/val
    num_train_img = train_img.shape[0] - num_val_img
    idx_rand = np.random.permutation(train_img.shape[0])
    train_img_new = train_img[idx_rand[:num_train_img], :, :, :]
    val_img = train_img[idx_rand[num_train_img:], :, :, :]
    train_label_new = train_label[idx_rand[:num_train_img]]
    val_label = train_label[idx_rand[num_train_img:]]

    CIFAR10_data = {}
    CIFAR10_data['train_img'] = train_img_new
    CIFAR10_data['val_img'] = val_img
    CIFAR10_data['test_img'] = test_img
    CIFAR10_data['train_label'] = train_label_new
    CIFAR10_data['val_label'] = val_label
    CIFAR10_data['test_label'] = test_label
    CIFAR10_data['mean_img'] = mean_img

    return CIFAR10_data


def read_CIFAR100(data_folder):
    """ Reads and parses examples from CIFAR100 python data files """

    # Constants describing the CIFAR-100 data set.
    img_height = 32
    img_width = 32
    num_class = 100
    num_channel = 3
    num_val_img = 5000

    train_img = []
    train_label = []
    test_img = []
    test_label = []

    train_file_list = ['train']
    test_file_list = ['test']

    tmp_dict = unpickle(os.path.join(data_folder, train_file_list[0]))
    train_img.append(tmp_dict['data'])
    train_label.append(tmp_dict['fine_labels'])

    tmp_dict = unpickle(os.path.join(data_folder, test_file_list[0]))
    test_img.append(tmp_dict['data'])
    test_label.append(tmp_dict['fine_labels'])

    train_img = np.concatenate(train_img)
    train_label = np.concatenate(train_label)
    test_img = np.concatenate(test_img)
    test_label = np.concatenate(test_label)

    train_img = np.reshape(train_img, [-1, num_channel, img_height, img_width])
    test_img = np.reshape(test_img, [-1, num_channel, img_height, img_width])

    # change format from [B, C, H, W] to [B, H, W, C] for feeding to Tensorflow
    train_img = np.transpose(train_img, [0, 2, 3, 1])
    test_img = np.transpose(test_img, [0, 2, 3, 1])

    mean_img = np.mean(np.concatenate([train_img, test_img]), axis=0)

    # random split for train/val
    num_train_img = train_img.shape[0] - num_val_img
    idx_rand = np.random.permutation(train_img.shape[0])
    train_img_new = train_img[idx_rand[:num_train_img], :, :, :]
    val_img = train_img[idx_rand[num_train_img:], :, :, :]
    train_label_new = train_label[idx_rand[:num_train_img]]
    val_label = train_label[idx_rand[num_train_img:]]

    CIFAR100_data = {}
    CIFAR100_data['train_img'] = train_img_new
    CIFAR100_data['val_img'] = val_img
    CIFAR100_data['test_img'] = test_img
    CIFAR100_data['train_label'] = train_label_new
    CIFAR100_data['val_label'] = val_label
    CIFAR100_data['test_label'] = test_label
    CIFAR100_data['mean_img'] = mean_img

    return CIFAR100_data
