import numpy as np
import cv2
import os
import sys

vgg_mean = [123.68, 116.779, 103.939]


def test_feature_extract(arg, Comparator, val_data, ch):
    test_features = np.array([]).reshape(0,1,1,512)
    test_pred_gender = []
    test_pred_race = []
    total_test_num = len(val_data)
    test_done = 0

    while total_test_num > 0:

        test_batch_size = min(arg.test_batch_size, total_test_num)

        if arg.dataset == 'Balanced':
            test_img_tmp = np.array(
                [np.subtract(cv2.resize(cv2.imread(os.path.join(arg.data_path, val_data['database'][b + test_done],
                                                                val_data['filename'][b + test_done]))[26:230, 26:230,
                                        :], (256, 256)),
                             vgg_mean) for b in range(test_batch_size)]) / 255.

        if arg.dataset == 'Morph':
            test_img_tmp = np.array(
                [np.subtract(cv2.resize(cv2.imread(os.path.join(arg.data_path, val_data['database'][b + test_done],
                                                                val_data['filename'][b + test_done])), (256, 256)),
                             vgg_mean) for b in range(test_batch_size)]) / 255.


        test_feature_tmp = Comparator.extract_features(test_img_tmp)
        test_features = np.concatenate([test_features, test_feature_tmp])

        if ch ==2 or ch == 6:
            gender_tmp = Comparator.pred_gender(test_img_tmp)
            test_pred_gender.extend(gender_tmp)
        if ch == 3 or ch ==6:
            race_tmp = Comparator.pred_race(test_img_tmp)
            test_pred_race.extend(race_tmp)

        total_test_num -= test_batch_size
        test_done += test_batch_size

        sys.stdout.write('\r')
        sys.stdout.write('| Test Feature Extract [%4d/%4d]' % (test_done, len(val_data)))
        sys.stdout.flush()

    return test_features, test_pred_gender,test_pred_race