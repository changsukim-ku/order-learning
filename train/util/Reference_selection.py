import numpy as np
import cv2
import os
import sys

vgg_mean = [123.68, 116.779, 103.939]

def ref_select(arg, Comparator, ref_data, oldest, ref_num, ch):

    if ch  == 1:
        gender_num = 1
        race_num = 1

    elif ch == 2:
        gender_num = 2
        race_num = 1

    elif ch == 3:
        gender_num = 1
        race_num = 3

    elif ch == 6:
        gender_num = 2
        race_num = 3

    ref_features = [[[] for _ in range(race_num)] for _ in range(gender_num)]
    ref_index = [[[] for _ in range(race_num)] for _ in range(gender_num)]
    ref_imgs = [[[] for _ in range(race_num)] for _ in range(gender_num)]

    ref_total_num = 0

    for ref_age in range(oldest + 1):
        for gender in range(gender_num):
            for race in range(race_num):

                ref = ref_data[(ref_data['age'] == ref_age)]

                if gender_num > 1:
                    ref = ref[(ref['gender'] == gender)]

                if race_num > 1:
                    ref = ref[(ref['race'] == race)]

                ref = ref.sort_values(by=['loss']).reset_index(drop=True)

                if len(ref) == 0:
                    ref_index[gender][race].append([])
                    ref_features[gender][race].append([])
                    ref_imgs[gender][race].append([])
                    continue
                if len(ref) > ref_num:
                    ref = ref[:ref_num]
                ref_index[gender][race].append(
                    [ref_data[ref_data['filename'] == ref['filename'][b]].index[0] for b in range(len(ref))])

                if arg.dataset == 'Balanced':
                    ref_img_tmp = np.array(
                        [np.subtract(cv2.resize(cv2.imread(os.path.join(arg.data_path, ref['database'][b],
                                                                        ref['filename'][b]))[26:230, 26:230, :],
                                                (256, 256)), vgg_mean)
                         for b in range(len(ref))]) / 255.

                if arg.dataset == 'Morph':
                    ref_img_tmp = np.array(
                        [np.subtract(cv2.resize(cv2.imread(os.path.join(arg.data_path, ref['database'][b],
                                                                        ref['filename'][b])),
                                                (256, 256)), vgg_mean)
                         for b in range(len(ref))]) / 255.


                ref_imgs[gender][race].append(ref_img_tmp)

                ref_feature_tmp = Comparator.extract_features(ref_img_tmp)
                ref_features[gender][race].append(ref_feature_tmp)

                ref_total_num += len(ref)

                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Age [%2d] Ref_num [%2d/%2d] Ref_total[%3d]' % (ref_age, len(ref), ref_num, ref_total_num))
                sys.stdout.flush()

    return ref_features, ref_index, ref_imgs


