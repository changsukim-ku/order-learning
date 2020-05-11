import numpy as np
import tensorflow as tf
from util.Comparator_model import Model_for_test
from util.Reference_selection import ref_select
from util.Train_feature_extraction import  train_feature_extract
from util.Predict_age import predict_age
from util.Make_random_pairs import make_datalist, load_batch
import os
import sys
import argparse
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0"
vgg_mean = [123.68, 116.779, 103.939]


def select_references(arg, train_data, epoch):


    initial_model = os.path.join(arg.save_dir, arg.dataset + '{:04d}.ckpt'.format(epoch+1))
    print(initial_model)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    tf.reset_default_graph()

    Comparator = Model_for_test(tfconfig=tfconfig, chain=arg.chain)
    Comparator.load_model(model_path=initial_model)


    print('Extract Features...')
    train_features =train_feature_extract(arg, Comparator, train_data)


    # Make pairs for training

    score_list = []
    for i in range(len(train_data)):

        test = train_data

        if arg.chain == 2:
            test = test[test['gender'] == train_data['gender'][i]]

        elif arg.chain == 3:
            test = test[test['race'] == train_data['race'][i]]

        elif arg.chain == 6:
            test = test[(test['gender'] == train_data['gender'][i]) & (test['race'] == train_data['race'][i])]

        greater = 1 * ((np.log(train_data.iloc[i]['age'])
                        - np.log(test['age'])) > arg.delta)
        less = 1 * ((np.log(train_data.iloc[i]['age'])
                     - np.log(test['age'])) < -arg.delta)
        equal = np.ones(len(test)) - greater - less
        labels = np.concatenate([np.expand_dims(greater, axis=1), np.expand_dims(equal, axis=1),
                                np.expand_dims(less, axis=1)], axis=1)

        score = Comparator.get_score(ref_features=np.repeat([train_features[i]], len(test), axis=0), test_features=train_features[test.index.tolist()], labels=labels)
        score_list.append(score)

        sys.stdout.write('\r')
        sys.stdout.write('| Get Score [%4d/%4d]' % (len(score_list), len(train_data)))
        sys.stdout.flush()

    #save
    train_data['loss'] = np.asarray(score_list)

    ref_name = arg.dataset + '{:04d}.txt'.format(epoch + 1)
    ckpt_path = arg.save_dir + '/' + ref_name

    train_data.to_csv(ckpt_path, sep=" ")


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
        args.save_dir = 'models/Balanced/%s' % args.experimental_setting

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

        # Select References
        for i in range(args.epoch, args.epoch - 10, -1):
            print('Epoch: %d' % (i))
            select_references(args, train_data=train_data, epoch=i)

    ############################################## MORPH Dataset #######################################################
    if args.dataset == 'Morph':

        for fold_num in range(5):

            # Fold Number
            args.fold = fold_num

            # Comparator Model
            args.save_dir = 'models/Morph/%s' % args.experimental_setting

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

            # Select References
            for i in range(args.epoch, args.epoch-10,-1):
                print('Epoch: %d'%(i))
                select_references(args, train_data= train_data, epoch=i)




