import os
import cv2
import numpy as np
import cPickle as pickle


def read_img_from_file(file_name, img_height, img_width):
    count = 0
    imgs = []
    labels = []

    with open(file_name) as f:
        for line in f:
            img_file, img_label = line.split()
            img = cv2.imread(img_file).astype(np.float32)
            # HWC -> WHC, compatible with caffe weights
            img = np.transpose(img, [1, 0, 2])
            img = cv2.resize(img, (img_width, img_height))

            imgs += [np.expand_dims(img, axis=0)]
            labels += [int(img_label)]
            count += 1

            if count % 1000 == 0:
                print 'Finish reading {:07d}'.format(count)

    return imgs, labels


def read_CUB(train_list_file, test_list_file):
    """ Reads and parses examples from CUB dataset """

    img_height = 227
    img_width = 227
    num_val_img = 500   # you can change the number of validation images here

    train_img = []
    train_label = []
    test_img = []
    test_label = []

    train_img, train_label = read_img_from_file(
        train_list_file, img_height, img_width)
    test_img, test_label = read_img_from_file(
        test_list_file, img_height, img_width)

    train_img = np.concatenate(train_img)
    test_img = np.concatenate(test_img)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    # random split for train/val
    num_train_img = train_img.shape[0] - num_val_img
    idx_rand = np.random.permutation(train_img.shape[0])
    train_img_new = train_img[idx_rand[:num_train_img], :, :, :]
    val_img = train_img[idx_rand[num_train_img:], :, :, :]
    train_label_new = train_label[idx_rand[:num_train_img]]
    val_label = train_label[idx_rand[num_train_img:]]

    CUB_data = {}
    CUB_data['train_img'] = train_img_new
    CUB_data['val_img'] = val_img
    CUB_data['test_img'] = test_img
    CUB_data['train_label'] = train_label_new
    CUB_data['val_label'] = val_label
    CUB_data['test_label'] = test_label

    return CUB_data


def compute_mean(train_list_file, test_list_file):
    CUB_data = read_CUB(train_list_file, test_list_file)
    mean_img = np.mean(np.concatenate(
        [CUB_data['train_img'], CUB_data['val_img'], CUB_data['test_img']]), axis=0)
    return mean_img


def save_mean_img(train_list_file, test_list_file, data_file='CUB_mean.p'):
    CUB_mean = compute_mean(train_list_file, test_list_file)
    pickle.dump(CUB_mean, open(data_file, 'wb'))
