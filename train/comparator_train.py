import numpy as np
import tensorflow as tf
from util.Comparator_model import Model_for_train
from util.Reference_selection import ref_select
from util.Test_feature_extraction import  test_feature_extract
from util.Predict_age import predict_age
from util.Make_random_pairs import make_datalist, load_batch
import os
import math
import argparse
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="1"
vgg_mean = [123.68, 116.779, 103.939]


def train(arg, train_data):


    initial_model = os.path.join(arg.initial_model_path)
    print(initial_model)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tf.reset_default_graph()

    Comparator = Model_for_train(tfconfig=tfconfig, learning_rate=1e-6, chain=arg.chain)
    Comparator.load_model(model_path=initial_model)

    writer = Comparator.make_log(log_path=arg.log_dir)

    print('Make Random Pairs...')

    # Make pairs for training


    step = 0

    for epoch in range(arg.epoch):

        train_a, train_b = make_datalist(arg, train_data, arg.chain)

        for batch in range(0, len(train_a)-arg.train_batch_size, arg.train_batch_size):
            print('Epoch: %d [%4d/%4d]' % ((epoch + 1), batch, len(train_a)-arg.train_batch_size))

            train_imgs_resize, train_age_gt, train_gender_gt, train_race_gt = load_batch(arg, train_a, train_b, vgg_mean, batch)
            summary = Comparator.train(train_imgs_resize, train_age_gt, train_gender_gt, train_race_gt)

            step += 1

            writer.add_summary(summary, global_step=step)

        #if (epoch+1) > (arg.epoch - 10):
        if (epoch + 1) == 1:
            Comparator.save_model(epoch, arg.dataset, arg.save_dir)


    Comparator.close_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=int, default=1, help = 'The number of chains')
    parser.add_argument('--delta', type=float, default=0.1, help = 'Set up the delta value')
    parser.add_argument('--epoch', type=int, default=80, help='The number of epoch')
    parser.add_argument('--experimental_setting', type=str, default='experimental_setting', help='Set up the experimental setting/ MORPH : A, B, C, D / Balanced : Supervised, Unsupervised')
    parser.add_argument('--dataset', type=str, default='dataset', help='Choose the dataset')
    parser.add_argument('--initial_model_path', type=str, default='ckpt', help='Initial model path for training')
    parser.add_argument('--save_dir', type=str, default='ckpt', help='Directory for saving a checkpoint file')
    parser.add_argument('--log_dir', type=str, default='log', help='Directory for saving a log file')
    parser.add_argument('--data_path', type=str, default='E:\Age_dataset', help = 'Data path of facial images')
    parser.add_argument('--train_batch_size', type=int, default=45, help = 'Train batch size')
    args = parser.parse_args()

    # Select Delta, Dataset, Experimental Setting
    args.delta = 0.10
    args.dataset = 'Morph'
    args.experimental_setting = 'A'
    args.chain = 1

    ############################################# Balacned Dataset #####################################################
    if args.dataset == 'Balanced':

        # Comparator Model
        args.initial_model_path = 'models/Pretrained/imdb_wiki.ckpt'
        args.save_dir = 'models/Balanced/%s' % args.experimental_setting
        args.log_dir = 'log/Balanced/%s' % args.experimental_setting

        # Train & Test Data
        train_data = pd.read_csv(os.path.join('index', 'Balanced', 'Balanced_train.txt'), sep=' ')
        val_data = pd.read_csv(os.path.join('index', 'Balanced', 'Balanced_test.txt'), sep=' ')

        print('Delta : ', args.delta)
        print('Experimental Setting : ', args.experimental_setting)
        print('Chain : %dCH' % args.chain)

        print('Train data len : ', len(train_data))
        print('Val data len : ', len(val_data))

        # Train
        train(args, train_data= train_data)

    ############################################## MORPH Dataset #######################################################
    if args.dataset == 'Morph':

        for fold_num in range(5):

            # Fold Number
            args.fold = fold_num
            # Comparator Model
            args.initial_model_path = 'models/Pretrained/imdb_wiki.ckpt'
            args.save_dir = 'models/Morph/%s' % args.experimental_setting
            args.log_dir = 'log/Morph/%s' % args.experimental_setting

            # Train & Test Data
            if args.experimental_setting == 'A':
                train_data = pd.read_csv(os.path.join('index', 'MORPH_Setting%s' % (args.experimental_setting),
                                                      'Setting%s_fold%d' % (
                                                          args.experimental_setting, args.fold) + '_train.txt'))
                val_data = pd.read_csv(os.path.join('index', 'MORPH_Setting%s' % (args.experimental_setting),
                                                    'Setting%s_fold%d' % (
                                                    args.experimental_setting, args.fold) + '_test.txt'))

            elif args.experimental_setting == 'B':
                fold_list = [0, 1, 2]
                fold_list.remove(args.fold)

                train_data = pd.read_csv(os.path.join('index', 'MORPH_Setting%s' % (args.experimental_setting),
                                                      'Setting%s_fold%d' % (args.experimental_setting, args.fold) + '.txt'),
                                         sep=" ")
                val_data_1 = pd.read_csv(os.path.join('index', 'MORPH_Setting%s' % (args.experimental_setting),
                                                      'Setting%s_fold%d' % (
                                                      args.experimental_setting, fold_list[0]) + '.txt'),
                                         sep=" ")
                val_data_2 = pd.read_csv(os.path.join('index', 'MORPH_Setting%s' % (args.experimental_setting),
                                                      'Setting%s_fold%d' % (
                                                      args.experimental_setting, fold_list[1]) + '.txt'),
                                         sep=" ")
                val_data = pd.concat([val_data_1, val_data_2]).reset_index(drop=True)

            else:
                data = pd.read_csv(os.path.join('index', 'MORPH_Setting%s' % (args.experimental_setting),
                                                'Setting%s' % (args.experimental_setting) + '.txt'), sep="\t")
                train_data = data.where(data['fold'] != args.fold)
                train_data = train_data.dropna(how='any')
                train_data = train_data.reset_index(drop=True)
                val_data = data.where(data['fold'] == args.fold)
                val_data = val_data.dropna(how='any')
                val_data = val_data.reset_index(drop=True)

            print('Experimental Setting : ', args.experimental_setting)

            print('Fold : ', args.fold)
            print('Delta : ', args.delta)

            print('Train data len : ', len(train_data))
            print('Val data len : ', len(val_data))

            # Train
            train(args, train_data= train_data)




