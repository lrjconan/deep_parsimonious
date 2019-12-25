# Learning Deep Parsimonious Representations

This is the code for our NIPS'16 paper:
* Renjie Liao, Alexander Schwing, Richard S. Zemel, Raquel Urtasun. [*Learning Deep Parsimonious Representations*](https://papers.nips.cc/paper/6263-learning-deep-parsimonious-representations).
Neural Information Processing System, 2016.

Please cite the above paper if you use our code.

The code is released under the [MIT license](LICENSE).

### Data

The configuration of data is as below,

* CIFAR10 and CIFAR100:

    1, Download data from https://www.cs.toronto.edu/~kriz/cifar.html

    2, Change the key `data_folder` in [exp_config.py](exp_config.py) as the unzipped path of data

* CUB-200-2011:

    1, Download data from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

    2, Preprocess (Crop + Resize + Subtract Mean) the images following the paper https://people.eecs.berkeley.edu/~nzhang/papers/eccv14_part.pdf

    3, Follow the example files [CUB_train_list.txt](CUB_train_list.txt) and [CUB_test_list.txt](CUB_test_list.txt) and specify the path of your own images
    
    4, Convert a pre-trained Alex-Net into Tensorflow format and specify the path in `caffe_model_file` of [exp_config.py](exp_config.py) (referring to `load_caffe_model` function in [AlexNet.py](AlexNet.py) to convert the model correctly)

    
### Training

Run `python run_train_model.py <exp_id>` to train a model.

Here exp_id should be one of the function names provided in [exp_config.py](exp_config.py). For example, setting `exp_id` to `CIFAR10_sample_clustering`, it will train a sample clustering model on CIFAR10 dataset.

### Testing

Run `python run_test_model.py <exp_id>` to test a model.

You need to specify the `test_model_name` and `test_folder` in [exp_config.py](exp_config.py) before run.

### Zero-Shot

Run `python run_zero_shot.py CUB_zero_shot` to train and test a Struct SVM on top of the learned feature.

We provide the train/val/test split used in our experiment as the file `zero_shot_split.npz`.

You need to specify the `test_model_name` and `test_folder` in [exp_config.py](exp_config.py) before run.

### Notes

* The experiment results may differ slightly from what we reported in the paper, as the
  cross validation is performed based on random split of the training data.
