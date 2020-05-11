import pandas as pd
import math
import numpy as np
import cv2
import os

def make_datalist(arg, data, ch):

    s_size = int(arg.train_batch_size / 1.5)
    d_size = arg.train_batch_size - s_size

    train_a = data.sample(frac=1).reset_index(drop=True)
    train_ad = train_a.iloc[int(len(data) * s_size / arg.train_batch_size):]
    train_ad = train_ad.reset_index(drop=True)
    train_as = train_a.iloc[:int(len(data) * s_size / arg.train_batch_size)]

    train_b = data.sample(frac=1).reset_index(drop=True)
    s_list = []
    for i in range(len(train_as)):
        age = train_as.iloc[i]['age']
        age_train_b = train_b.loc[(abs(math.log(age) - np.log(train_b['age'])) <= arg.delta + 0.1)]
        if ch == 2 or ch == 6:
            age_train_b = age_train_b.loc[(train_as.iloc[i]['gender'] == age_train_b['gender'])]
        if ch == 3 or ch == 6:
            age_train_b = age_train_b.loc[(train_as.iloc[i]['race'] == age_train_b['race'])]
        newidx = age_train_b.sample(n=1)
        s_list.append(newidx.index[0])
    train_bs = train_b.loc[s_list]
    train_bs = train_bs.reset_index(drop=True)

    train_bd = train_b.drop(s_list)
    train_bd = train_bd.reset_index(drop=True)
    d_list = []
    for i in range(len(train_ad)):
        age_train_b = train_bd
        if ch == 2 or ch == 6:
            age_train_b = age_train_b.loc[(age_train_b['gender'] == train_ad.iloc[i]['gender'])]
        if ch == 3 or ch == 6:
            age_train_b = age_train_b.loc[(age_train_b['race'] == train_ad.iloc[i]['race'])]
        newidx = age_train_b.sample(n=1)
        d_list.append(newidx.index[0])
    train_bd = train_bd.loc[d_list]
    train_bd = train_bd.reset_index(drop=True)

    train_a = pd.Series([])
    train_b = pd.Series([])
    for i in range(0, len(data) - arg.train_batch_size, arg.train_batch_size):
        j = i / arg.train_batch_size
        train_a = pd.concat([train_a, train_as.loc[(j * s_size):((j + 1) * s_size - 1)]], ignore_index=True,
                            sort=False)
        train_a = pd.concat([train_a, train_ad.loc[(j * d_size):((j + 1) * d_size - 1)]], ignore_index=True,
                            sort=False)
        train_b = pd.concat([train_b, train_bs.loc[(j * s_size):((j + 1) * s_size - 1)]], ignore_index=True,
                            sort=False)
        train_b = pd.concat([train_b, train_bd.loc[(j * d_size):((j + 1) * d_size - 1)]], ignore_index=True,
                            sort=False)
    return train_a, train_b

def load_batch(arg, train_a, train_b, mean_value, batch_num):#Balnced [26:230, 26:230, :] 추가하기

    if arg.dataset == 'Balanced':

        train_imgs_resize_a = np.array([np.subtract(cv2.resize(cv2.imread(
            os.path.join(arg.data_path, train_a.iloc[batch_num + img]['database'],
                         train_a.iloc[batch_num + img]['filename']))[26:230, 26:230, :],
            (256, 256)), mean_value)
            for img in range(arg.train_batch_size)], dtype=np.float32) / 255.  #

        train_imgs_resize_b = np.array([np.subtract(cv2.resize(cv2.imread(
            os.path.join(arg.data_path, train_b.iloc[batch_num + img]['database'],
                         train_b.iloc[batch_num + img]['filename']))[26:230, 26:230, :],
            (256, 256)), mean_value)
            for img in range(arg.train_batch_size)], dtype=np.float32) / 255.  #

    if arg.dataset == 'Morph':
        train_imgs_resize_a = np.array([np.subtract(cv2.resize(cv2.imread(
            os.path.join(arg.data_path, train_a.iloc[batch_num + img]['database'],
                         train_a.iloc[batch_num + img]['filename'])),
            (256, 256)), mean_value)
            for img in range(arg.train_batch_size)], dtype=np.float32) / 255.  #

        train_imgs_resize_b = np.array([np.subtract(cv2.resize(cv2.imread(
            os.path.join(arg.data_path, train_b.iloc[batch_num + img]['database'],
                         train_b.iloc[batch_num + img]['filename'])),
            (256, 256)), mean_value)
            for img in range(arg.train_batch_size)], dtype=np.float32) / 255.  #

    train_imgs_resize = np.concatenate([train_imgs_resize_a, train_imgs_resize_b], axis=0)

    train_age_gt = np.array(
        [[0., 0., 1.] if np.log(train_a.iloc[batch_num + label]['age'])
                         < np.log(train_b.iloc[batch_num + label]['age']) - arg.delta
         else [1., 0., 0.] if np.log(train_a.iloc[batch_num + label]['age'])
                              > np.log(train_b.iloc[batch_num + label]['age']) + arg.delta
        else [0., 1., 0.]
         for label in range(arg.train_batch_size)], dtype=np.float32)

    train_gender = np.array([train_a.iloc[batch_num + img]['gender'] for img in range(arg.train_batch_size)])

    train_gender_gt = np.array([[1., 0.] if train_gender[label] == 0
                                else [0., 1.] for label in range(len(train_gender))])

    train_race = np.array([train_a.iloc[batch_num + img]['race'] for img in range(arg.train_batch_size)])

    train_race_gt = np.array([[1., 0., 0.] if train_race[label] == 0
                            else [0., 1., 0.] if train_race[label] == 1
                                else [0., 0., 1.] for label in range(len(train_race))])

    return train_imgs_resize, train_age_gt, train_gender_gt, train_race_gt

