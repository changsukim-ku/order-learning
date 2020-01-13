import numpy as np
import tensorflow as tf
from util.Comparator_model import Model
from util.Reference_selection import ref_select
from util.Test_feature_extraction import  test_feature_extract
from util.Predict_age import predict_age
import os
import math
import argparse
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0"
vgg_mean = [123.68, 116.779, 103.939]


def Age_Prediction(arg, val_data, ref_data, ref_num=5):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tf.reset_default_graph()

    test_model = os.path.join(arg.ckpt_dir,  arg.test_model_path)
    print(test_model)

    Comparator = Model(chain=arg.chain)
    Comparator.load_model(model_path=test_model)

    print('Extract Reference...')

    # Select Best References
    ref_data_sort_by_age = ref_data.sort_values(by=['age']).reset_index(drop=True)

    youngest = int(ref_data_sort_by_age['age'][0])
    oldest = int(ref_data_sort_by_age['age'][len(ref_data)-1])

    ref_features, ref_index, ref_imgs = ref_select(arg, Comparator, ref_data, oldest, ref_num, arg.chain)

    # Extract Test Images Features

    print('\nExtract Test Features...')

    test_features, test_pred_gender, test_pred_race = test_feature_extract(arg, Comparator, val_data, arg.chain)

    if arg.chain == 2 or arg.chain == 6:
        val_data['pred_gender'] = test_pred_gender

    if arg.chain == 3 or arg.chain == 6:
        val_data['pred_race'] = test_pred_race


    # Age Prediction

    print('\nAge Prediction...')

    lb_list = []
    ub_list = []
    for age_tmp in range(81):
        lb_list.append(sum(np.arange(81) < age_tmp * math.exp(-arg.delta)))
        ub_list.append(sum(np.arange(81) < age_tmp * math.exp(arg.delta)))

    pred_age = predict_age(arg.chain, youngest, oldest, lb_list, ub_list, ref_features, test_features, val_data, Comparator)

    Error = np.mean(np.abs(np.subtract(val_data['age'] , pred_age)))
    CS = np.sum(np.abs(np.array(pred_age) - val_data['age']) <= 5) / float(len(val_data))
    print("\nError: %.2f\tCS:%.4f"%(Error,CS))

    Comparator.close_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=float, default=0, help = 'The number of chains')
    parser.add_argument('--delta', type=float, default=0, help = 'Set up the delta value')
    parser.add_argument('--experimental_setting', type=str, default='experimental_setting', help='Set up the experimental setting/ MORPH : A, B, C, D / Balanced : Supervised, Unsupervised')
    parser.add_argument('--dataset', type=str, default='dataset', help='Choose the dataset')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='Parent directory for the checkpoint file')
    parser.add_argument('--test_model_path', type=str, default='test', help='Checkpoint file name')
    parser.add_argument('--ref_path', type=str, default='ref', help='Reference data path')
    parser.add_argument('--data_path', type=str, default='E:\Age_dataset', help = 'Data path of facial images')
    parser.add_argument('--test_batch_size', type=int, default=150, help = 'Test batch size')
    args = parser.parse_args()

    # Select Delta, Dataset, Experimental Setting
    args.delta = 0.10
    args.dataset = 'Balanced'
    args.experimental_setting = 'Supervised'
    args.chain = 1

    ############################################# Balacned Dataset #####################################################
    if args.dataset == 'Balanced':

        # Comparator Model
        args.ckpt_dir = 'models/Balanced/%s' % args.experimental_setting
        args.test_model_path = '%dCH/%dCH.ckpt' % (args.chain, args.chain)

        # Reference Data
        args.ref_path = '%dCH/%dCH' % (args.chain, args.chain)
        ref_data = pd.read_csv(os.path.join(args.ckpt_dir, args.ref_path + '_ref_data.txt'), sep=' ')

        # Train & Test Data
        train_data = pd.read_csv(os.path.join('index', 'Balanced', 'Balanced_train.txt'), sep=' ')
        val_data = pd.read_csv(os.path.join('index', 'Balanced', 'Balanced_test.txt'), sep=' ')

        print('Delta : ', args.delta)
        print('Experimental Setting : ', args.experimental_setting)
        print('Chain : %dCH' % args.chain)

        print('Train data len : ', len(train_data))
        print('Val data len : ', len(val_data))

        # Test
        Age_Prediction(args, val_data=val_data, ref_data=ref_data, ref_num=5)

    ############################################## MORPH Dataset #######################################################
    if args.dataset == 'Morph':

        for fold_num in range(5):

            # Fold Number
            args.fold = fold_num
            # Comparator Model
            args.ckpt_dir = 'models/MORPH_Setting%s' % (args.experimental_setting)
            args.test_model_path = 'Fold%d.ckpt' % (args.fold)
            # Reference Data
            args.ref_path = 'Fold%d' % (args.fold)
            ref_data = pd.read_csv(os.path.join(args.ckpt_dir, args.ref_path + '_ref_data.txt'), sep=' ')
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

            # Test
            Age_Prediction(args, val_data=val_data, ref_data=ref_data, ref_num=5)




