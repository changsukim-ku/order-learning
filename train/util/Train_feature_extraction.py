import numpy as np
import cv2
import os
import sys

vgg_mean = [123.68, 116.779, 103.939]


def train_feature_extract(arg, Comparator, train_data):
    train_features = np.array([]).reshape(0,1,1,512)
    total_train_num = len(train_data)
    train_done = 0

    while total_train_num > 0:

        train_batch_size = min(arg.train_batch_size, total_train_num)

        if arg.dataset == 'Balanced':
            train_img_tmp = np.array(
                [np.subtract(cv2.resize(cv2.imread(os.path.join(arg.data_path, train_data['database'][b + train_done],
                                                                train_data['filename'][b + train_done]))[26:230, 26:230,
                                        :], (256, 256)),
                             vgg_mean) for b in range(train_batch_size)]) / 255.

        if arg.dataset == 'Morph':
            train_img_tmp = np.array(
                [np.subtract(cv2.resize(cv2.imread(os.path.join(arg.data_path, train_data['database'][b + train_done],
                                                                train_data['filename'][b + train_done])), (256, 256)),
                             vgg_mean) for b in range(train_batch_size)]) / 255.


        train_feature_tmp = Comparator.extract_features(train_img_tmp)
        train_features = np.concatenate([train_features, train_feature_tmp])

        total_train_num -= train_batch_size
        train_done += train_batch_size

        sys.stdout.write('\r')
        sys.stdout.write('| Train Feature Extract [%4d/%4d]' % (train_done, len(train_data)))
        sys.stdout.flush()

    return train_features