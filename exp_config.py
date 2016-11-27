""" parameters of experiments """


def CIFAR10_baseline():
    param = {
        'device': '/gpu:0',
        'data_folder': 'CIFAR10/cifar-10-batches-py',  # the path of unzipped CIFAR10 data
        'save_path': '',  # the path to save your model
        'dataset_name': 'CIFAR10',
        'model_name': 'baseline',
        'merge_valid': False,
        'resume_training': False,
        'bat_size': 100,
        'img_height': 32,
        'img_width': 32,
        'img_channel': 3,
        'disp_iter': 100,
        'save_iter': 10000,
        'max_train_iter': 100000,
        'valid_iter': 1000,
        'base_learn_rate': 1.0e-2,
        'learn_rate_decay_step': 2000,
        'learn_rate_decay_rate': 0.85,
        'label_size': 10,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'init_std_cnn': [1.0e-2, 1.0e-2, 1.0e-2],
        'init_std_mlp': [1.0e-1, 1.0e-1],
        'filter_shape': [[5, 5, 3, 32], [5, 5, 32, 32], [5, 5, 32, 64]],
        'filter_stride': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        'pool_func': ['max', 'avg', 'avg'],
        'pool_size': [[1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1]],
        'pool_stride': [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
        'act_func_cnn': ['relu'] * 3,
        'act_func_mlp': [None] * 2,
        'dims_mlp': [64, 10, 1024]
    }

    return param


def CIFAR100_baseline():
    param = CIFAR10_baseline()
    param['dataset_name'] = 'CIFAR100'
    param['data_folder'] = ''  # the path of unzipped CIFAR100 data
    param['label_size'] = 100
    param['init_std_cnn'] = [1.0e-1, 1.0e-1, 1.0e-1]
    param['dims_mlp'] = [64, 100, 1024]

    return param


def CIFAR10_sample_clustering():
    param = CIFAR10_baseline()
    param['eta'] = 0.1
    param['model_name'] = 'parsimonious'
    param['init_std_cnn'] = [1.0e-2, 1.0e-2, 1.0e-2]
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['sample', 'sample', 'sample']
    param['clustering_shape_cnn'] = [[100, 32768], [100, 8192], [100, 4096]]
    param['clustering_alpha_cnn'] = [1.0e-1, 1.0e-1, 1.0e-1]
    param['num_cluster_mlp'] = [100, 100]
    param['clustering_shape_mlp'] = [[100, 64], [100, 10]]
    param['clustering_alpha_mlp'] = [1.0e-1, 1.0e-1]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = ''  # the path of your testing model
    param['resume_training'] = False

    if param['resume_training'] == True:
        param['max_train_iter'] = 60000
        param['base_learn_rate'] = 1.0e-3
        param['merge_valid'] = True
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''  # exp id of resume
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR10_spatial_clustering():
    param = CIFAR10_baseline()
    param['eta'] = 0.1
    param['model_name'] = 'parsimonious'
    param['init_std_cnn'] = [1.0e-2, 1.0e-2, 1.0e-2]
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['spatial', 'spatial', 'spatial']
    param['clustering_shape_cnn'] = [[102400, 32], [25600, 32], [6400, 64]]
    param['clustering_alpha_cnn'] = [1.0e-1, 1.0e-1, 1.0e-1]
    param['num_cluster_mlp'] = [100, 100]
    param['clustering_shape_mlp'] = [[100, 64], [100, 10]]
    param['clustering_alpha_mlp'] = [1.0e-1, 1.0e-1]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training'] == True:
        param['max_train_iter'] = 60000
        param['base_learn_rate'] = 1.0e-3
        param['merge_valid'] = True
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR10_channel_clustering():
    param = CIFAR10_baseline()
    param['eta'] = 0.1
    param['model_name'] = 'parsimonious'
    param['init_std_cnn'] = [1.0e-2, 1.0e-2, 1.0e-2]
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['channel', 'channel', 'channel']
    param['clustering_shape_cnn'] = [[3200, 1024], [3200, 256], [6400, 64]]
    param['clustering_alpha_cnn'] = [1.0e-1, 1.0e-1, 1.0e-1]
    param['num_cluster_mlp'] = [100, 100]
    param['clustering_shape_mlp'] = [[100, 64], [100, 10]]
    param['clustering_alpha_mlp'] = [1.0e-1, 1.0e-1]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training'] == True:
        param['max_train_iter'] = 60000
        param['base_learn_rate'] = 1.0e-3
        param['merge_valid'] = True
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR100_sample_clustering():
    param = CIFAR100_baseline()
    param['eta'] = 0.05
    param['model_name'] = 'parsimonious'
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['sample', 'sample', 'sample']
    param['clustering_shape_cnn'] = [[100, 32768], [100, 8192], [100, 4096]]
    param['clustering_alpha_cnn'] = [0.0e+0, 0.0e+0, 0.0e+0]
    param['num_cluster_mlp'] = [100, None]
    param['clustering_shape_mlp'] = [[100, 64], None]
    param['clustering_alpha_mlp'] = [0.0e+0, None]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0050000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training'] == True:
        param['max_train_iter'] = 60000
        param['merge_valid'] = True
        param['base_learn_rate'] = 1.0e-3
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR100_spatial_clustering():
    param = CIFAR100_baseline()
    param['eta'] = 0.05
    param['model_name'] = 'parsimonious'
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['spatial', 'spatial', 'spatial']
    param['clustering_shape_cnn'] = [[102400, 32], [25600, 32], [6400, 64]]
    param['clustering_alpha_cnn'] = [1.0e+1, 1.0e+0, 1.0e+0]
    param['num_cluster_mlp'] = [100, None]
    param['clustering_shape_mlp'] = [[100, 64], None]
    param['clustering_alpha_mlp'] = [1.0e+0, None]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training'] == True:
        param['max_train_iter'] = 60000
        param['merge_valid'] = True
        param['base_learn_rate'] = 1.0e-3
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CIFAR100_channel_clustering():
    param = CIFAR100_baseline()
    param['eta'] = 0.05
    param['model_name'] = 'parsimonious'
    param['num_cluster_cnn'] = [100, 100, 100]
    param['clustering_type_cnn'] = ['channel', 'channel', 'channel']
    param['clustering_shape_cnn'] = [[3200, 1024], [3200, 256], [6400, 64]]
    param['clustering_alpha_cnn'] = [1.0e+1, 1.0e+0, 1.0e+0]
    param['num_cluster_mlp'] = [100, None]
    param['clustering_shape_mlp'] = [[100, 64], None]
    param['clustering_alpha_mlp'] = [1.0e+0, None]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0060000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training'] == True:
        param['max_train_iter'] = 60000
        param['merge_valid'] = True
        param['base_learn_rate'] = 1.0e-3
        param['resume_step'] = 50000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0050000.ckpt'

    return param


def CUB_baseline():
    param = {
        'device': '/gpu:0',
        # path of mean image, you can use save_mean_img function in CUB_input.py
        'mean_img': 'CUB_mean.p',
        # caffe model file
        'caffe_model_file': '',
        'train_list_file': 'CUB_train_list.txt',
        'test_list_file': 'CUB_test_list.txt',
        'save_path': '',
        'dataset_name': 'CUB',
        'model_name': 'baseline',
        'merge_valid': False,
        'resume_training': False,
        'bat_size': 256,
        'img_height': 227,
        'img_width': 227,
        'img_channel': 3,
        'disp_iter': 10,
        'save_iter': 1000,
        'max_train_iter': 2000,
        'valid_iter': 100,
        'base_learn_rate': 1.0e-3,
        'learn_rate_decay_step': 1000,
        'learn_rate_decay_rate': 0.1,
        'label_size': 200,
        'momentum': 0.9,
        'dropout_rate': 0.5,
        'weight_decay': 0.0,
        'using_caffe_weights': True,
        'init_std': [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 5.0e-3, 5.0e-3, 1.0e-2],
        'init_bias': [0.0, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0],
        'filter_shape': [[11, 11, 3, 96], [5, 5, 48, 256], [3, 3, 256, 384], [3, 3, 192, 384], [3, 3, 192, 256], [9216, 4096], [4096, 4096], [4096, 200]]
    }

    return param

# AlexNet's feature map size = 55 x 55 x 96, 27 x 27 x 256, 13 x 13 x 384,
# 13 x 13 x 384, 13 x 13 x 256


def CUB_sample_clustering():
    param = CUB_baseline()
    param['model_name'] = 'parsimonious'
    param['eta'] = 0.5
    param['num_cluster'] = [200, 200, 200, 200, 200, 200, 200, None]
    param['clustering_type'] = ['sample', 'sample',
                                'sample', 'sample', 'sample', None, None, None]
    param['clustering_shape'] = [[256, 290400], [256, 186624], [256, 64896], [
        256, 64896], [256, 43264], [256, 4096], [256, 4096], None]
    param['clustering_alpha'] = [1.0e-5, 1.0e-5,
                                 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-3, 1.0e-3, None]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0003000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training'] == True:
        param['max_train_iter'] = 3000
        param['base_learn_rate'] = 1.0e-5
        param['merge_valid'] = True
        param['resume_step'] = 2000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0002000.ckpt'
    return param


def CUB_spatial_clustering():
    param = CUB_baseline()
    param['model_name'] = 'parsimonious'
    param['eta'] = 0.5
    param['num_cluster'] = [200, 200, 200, 200, 200, 200, 200, None]
    param['clustering_type'] = ['spatial', 'spatial',
                                'spatial', 'spatial', 'spatial', None, None, None]
    param['clustering_shape'] = [[774400, 96], [186624, 256], [43264, 384], [
        43264, 384], [43264, 256], [256, 4096], [256, 4096], None]
    param['clustering_alpha'] = [1.0e-5, 1.0e-5,
                                 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-3, 1.0e-3, None]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0003000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training'] == True:
        param['max_train_iter'] = 3000
        param['base_learn_rate'] = 1.0e-5
        param['merge_valid'] = True
        param['resume_step'] = 2000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0002000.ckpt'
    return param


def CUB_channel_clustering():
    param = CUB_baseline()
    param['model_name'] = 'parsimonious'
    param['eta'] = 0.5
    param['num_cluster'] = [200, 200, 200, 200, 200, 200, 200, None]
    param['clustering_type'] = ['channel', 'channel',
                                'channel', 'channel', 'channel', None, None, None]
    param['clustering_shape'] = [[24576, 3025], [65536, 729], [98304, 169], [
        98304, 169], [65536, 169], [256, 4096], [256, 4096], None]
    param['clustering_alpha'] = [1.0e-5, 1.0e-5,
                                 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-3, 1.0e-3, None]
    param['clustering_iter'] = 1
    param['test_model_name'] = 'parsimonious_snapshot_0003000.ckpt'
    param['test_folder'] = ''
    param['resume_training'] = False

    if param['resume_training'] == True:
        param['max_train_iter'] = 3000
        param['base_learn_rate'] = 1.0e-5
        param['merge_valid'] = True
        param['resume_step'] = 2000
        param['resume_exp_id'] = ''
        param['resume_model_name'] = 'parsimonious_snapshot_0002000.ckpt'
    return param


def CUB_zero_shot():
    param = {
        # this file is provided by CUB dataset, change the path to yours
        'embedding_file': 'CUB_200_2011/attributes/class_attribute_labels_continuous.txt',
        'load_feat': False,
        'run_test': True,
        'feat_file': 'feat.npz',
        'split_file': 'zero_shot_split.npz',
        # neural network parameters
        'model_config': CUB_baseline(),
        # struct svm parameters
        'ssvm_param': {
            'eta': 1.0e+1,
            'batch_size': 256,
            'num_train_iter': 80,
            'num_train_imgs': 5894,
            'num_val_imgs': 2961,
            'num_test_imgs': 2933,
            'lr_decay_rate': 0.5,
            'lr_decay_iter': 40,
            'val_iter': 20
        }
    }

    # put the path of learned NN model here
    param['model_config']['test_folder'] = ''
    # put the name of learned NN model here
    param['model_config']['test_model_name'] = ''

    return param
