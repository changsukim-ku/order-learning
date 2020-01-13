import numpy as np
import sys


def predict_age(ch, youngest, oldest, lb_list, ub_list, ref_features, test_features, val_data, Comparator):

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

    pred_age = []
    total_test_num = len(val_data)
    test_done = 0

    while total_test_num > 0:

        test_batch_size = min(len(val_data), total_test_num)

        compare_matrix = np.zeros([test_batch_size, ub_list[-1]])

        for gender in range(gender_num):
            for race in range(race_num):

                test_features_filtered_index = val_data

                if ch == 2 or ch == 3:
                    test_features_filtered_index = test_features_filtered_index[(test_features_filtered_index['pred_gender']==gender)]

                if ch == 3 or ch == 6:
                    test_features_filtered_index = test_features_filtered_index[(test_features_filtered_index['pred_race'] == race)]

                test_features_filtered_index = test_features_filtered_index.index.tolist()
                test_features_filtered = test_features[test_features_filtered_index]

                for theta1 in range(youngest, oldest+1):

                    ref_feature_tmp = np.array(
                        [np.repeat(np.array([ref_features[gender][race][theta1][b]]), len(test_features_filtered), axis=0) for b in
                         range(len(ref_features[gender][race][theta1]))])

                    if len(ref_features[gender][race][theta1]) == 0:
                        continue

                    for i in range(len(ref_features[gender][race][theta1])):
                        comparator_results = Comparator.pred_order(ref_vectors_batch=ref_feature_tmp[i], test_vectors_batch= test_features_filtered)
                        compare_matrix[test_features_filtered_index, :lb_list[theta1]] += np.tile(comparator_results[:, 0],
                                                                                                  (lb_list[theta1], 1)).transpose()
                        compare_matrix[test_features_filtered_index, lb_list[theta1]:ub_list[theta1]] += np.tile(comparator_results[:, 1], ((ub_list[theta1] - lb_list[theta1]), 1)).transpose()
                        compare_matrix[test_features_filtered_index, ub_list[theta1]:] += np.tile(comparator_results[:, 2], (
                            ub_list[-1] - ub_list[theta1], 1)).transpose()

                    sys.stdout.write('\r')
                    sys.stdout.write('| Gender [%d] Race [%d] Compare with Reference Images [%2d/%2d]' % (gender, race, theta1, oldest))
                    sys.stdout.flush()


        pred_age_tmp = [int(np.mean(np.where(compare_matrix[b]==np.max(compare_matrix[b])))+0.5) for b in range(test_batch_size)]
        pred_age.extend(pred_age_tmp)

        total_test_num -= test_batch_size
        test_done += test_batch_size

    return pred_age