from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import util.vgg as vgg
from util.ckpt_function import *
import tensorflow.contrib.slim as slim
import tensorflow as tf



def conv1x1(x, outplanes, activation_fn, name):
    with arg_scope([layers.conv2d], trainable=False):
        x = layers.conv2d(x, outplanes, [1, 1], activation_fn=activation_fn, scope=name)
    return x

def dropout(x, name, prob = 0.8):
    with arg_scope([layers.dropout], is_training=False):
        x = layers.dropout(x, prob, scope=name)
    return x

class Model_for_train:
    def __init__(self, tfconfig, learning_rate, chain):

        self.chain = chain
        self.learning_rate = learning_rate

        self.imgs = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='image')
        self.labels = tf.placeholder(tf.int32, shape=[None, 3], name='labels')
        self.labels_gender = tf.placeholder(tf.int32, shape=[None, 2], name='labels_gender')
        self.labels_race = tf.placeholder(tf.int32, shape=[None, 3], name='labels_race')


        images_aug = tf.map_fn(lambda image: tf.random_crop(image, (224, 224,3)), self.imgs,
                               name='crop_image')
        images_aug = tf.map_fn(lambda image: tf.image.random_flip_left_right(image), images_aug,
                               name='flip_image')
        images_aug = tf.image.random_brightness(images_aug, max_delta=32. / 255.)
        images_aug = tf.image.random_saturation(images_aug, lower=0.5, upper=1.5)
        images_aug = tf.image.random_hue(images_aug, max_delta=0.2)
        images_aug = tf.image.random_contrast(images_aug, lower=0.5, upper=1.5)

        images_a, images_b = tf.split(images_aug, num_or_size_splits=2, axis=0)



        self.sess = tf.Session(config=tfconfig)

        with slim.arg_scope(vgg.vgg_arg_scope()):
            self.feature_a, _ = vgg.vgg_16(images_a, num_classes=0, is_training=False, fc_layers=False,
                                         spatial_squeeze=False, global_pool=True, scope='vgg_16')
            self.feature_b, _ = vgg.vgg_16(images_b, num_classes=0, is_training=False, fc_layers=False, reuse=True,
                                         spatial_squeeze=False, global_pool=True, scope='vgg_16')

            if self.chain > 1:

                case = self.feature_a
                case = conv1x1(case, outplanes=512, activation_fn=nn_ops.relu, name='fc1_case')
                case = dropout(case, name='do1_case')
                case = conv1x1(case, outplanes=512, activation_fn=nn_ops.relu, name='fc2_case')
                case = dropout(case, name='do2_case')

                if self.chain == 2 or self.chain == 6:
                    g_logits = conv1x1(case, outplanes=2, activation_fn=None, name='fc3_gender')
                    self.g_logits = array_ops.squeeze(g_logits, [1, 2], name='SpatialSqueeze')

                if self.chain == 3 or self.chain == 6:
                    r_logits = conv1x1(case, outplanes=3, activation_fn=None, name='fc3_race')
                    self.r_logits = array_ops.squeeze(r_logits, [1, 2], name='SpatialSqueeze')

            feature_cat = tf.concat([self.feature_a, self.feature_b], axis=3)
            feature_cat = conv1x1(feature_cat, outplanes=512, activation_fn=nn_ops.relu, name='fc1')
            feature_cat = dropout(feature_cat, name='do1')
            feature_cat = conv1x1(feature_cat, outplanes=512, activation_fn=nn_ops.relu, name='fc2')
            feature_cat = dropout(feature_cat, name='do2')
            f_logits = conv1x1(feature_cat, outplanes=3, activation_fn=None, name='fc3')
            f_logits = array_ops.squeeze(f_logits, [1, 2], name='SpatialSqueeze')
            self.f_logits_softmax = slim.softmax(f_logits)

            self.loss_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=f_logits, labels=self.labels))
            with tf.name_scope('training_loss'):
                tf.summary.scalar('comparator_loss', self.loss_softmax)
            loss_total = self.loss_softmax

            if self.chain == 2 or self.chain == 6:
                self.loss_softmax_gender = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=g_logits, labels=self.labels_gender))
                with tf.name_scope('training_loss'):
                    tf.summary.scalar('gender_loss', self.loss_softmax_gender)
                loss_total += self.loss_softmax_gender

            if self.chain == 3 or self.chain == 6:
                self.loss_softmax_race = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=r_logits, labels=self.labels_race))
                with tf.name_scope('training_loss'):
                    tf.summary.scalar('race_loss', self.loss_softmax_race)
                loss_total += self.loss_softmax_race


            # optimize
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # batch_normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_op = slim.learning.create_train_op(loss_total, optimizer)

            with tf.name_scope('training_accuracy'):
                correct_prediction = tf.equal(tf.argmax(f_logits, 1), tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

                if self.chain == 2 or self.chain == 6:
                    correct_prediction_gender = tf.equal(tf.argmax(g_logits, 1), tf.argmax(self.labels_gender, 1))
                    self.accuracy_gender = tf.reduce_mean(tf.cast(correct_prediction_gender, tf.float32))
                    tf.summary.scalar('accuracy_gender', self.accuracy_gender)

                if self.chain == 3 or self.chain == 6:
                    correct_prediction_race = tf.equal(tf.argmax(r_logits, 1), tf.argmax(self.labels_race, 1))
                    self.accuracy_race = tf.reduce_mean(tf.cast(correct_prediction_race, tf.float32))
                    tf.summary.scalar('accuracy_race', self.accuracy_race)

            self.merged_summary_op = tf.summary.merge_all()

            self.saver = tf.train.Saver(max_to_keep=50)


    def make_log(self, log_path):

        writer = tf.summary.FileWriter(log_path, self.sess.graph)

        return writer

    def load_model(self, model_path):
        print('Load Model...')

        variables = tf.global_variables()
        var_keep_dic = get_variables_in_checkpoint(model_path)
        variables_to_restore = get_variables_to_restore(variables, var_keep_dic)

        self.sess.run(tf.variables_initializer(variables, name='init'))
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(self.sess, model_path)

        print('Load Finish...')

    def save_model(self, epoch, ckpt_name, save_path):
        print('Save Model...')

        ckpt_name = ckpt_name +'{:04d}.ckpt'.format(epoch + 1)
        ckpt_path = save_path + '/' + ckpt_name
        self.saver.save(self.sess, ckpt_path)

        print('Saved Model in %s...'%(ckpt_path))


    def train(self, imgs, age_gt, gender_gt, race_gt):

        feed_dict = {self.imgs: imgs, self.labels: age_gt, self.labels_gender: gender_gt, self.labels_race: race_gt}

        if self.chain == 1:
            acc,summary, loss_com, _ = \
                self.sess.run([self.accuracy, self.merged_summary_op, self.loss_softmax, self.train_op],
                feed_dict=feed_dict)
            print('Comparator Acc: %.4f, Comparator Loss: %.4f'%(acc, loss_com))

        if self.chain == 2:
            acc, acc_gender, summary, loss_com, loss_gender, _ = \
                self.sess.run([self.accuracy, self.accuracy_gender, self.merged_summary_op, self.loss_softmax,
                 self.loss_softmax_gender, self.train_op],
                feed_dict=feed_dict)
            print('Comparator Acc: %.4f, Gender Classifier Acc: %.4f, '
                  'Comparator Loss: %.4f, Gender Classifier Loss: %.4f' % (acc, acc_gender, loss_com, loss_gender))

        if self.chain == 3:
            acc, acc_race, summary, loss_com, loss_race, _ = \
                self.sess.run([self.accuracy, self.accuracy_race, self.merged_summary_op, self.loss_softmax,
                self.loss_softmax_race, self.train_op],
                feed_dict=feed_dict)
            print('Comparator Acc: %.4f, Race Classifier Acc: %.4f, '
                  'Comparator Loss: %.4f, RaceGender Classifier Loss: %.4f' % (acc, acc_race, loss_com, loss_race))

        if self.chain == 6:
            acc, acc_gender, acc_race, summary, loss_com, loss_gender, loss_race, _ = \
                self.sess.run([self.accuracy, self.accuracy_gender, self.accuracy_race, self.merged_summary_op, self.loss_softmax,
                 self.loss_softmax_gender, self.loss_softmax_race, self.train_op],
                feed_dict=feed_dict)
            print('Comparator Acc: %.4f, Gender Classifier Acc: %.4f, Race Classifier Acc: %.4f, '
                  'Comparator Loss: %.4f, Gender Classifier Loss: %.4f, Race Classifier Loss: %.4f' % (
                  acc, acc_gender, acc_race, loss_com, loss_gender, loss_race))

        return summary


    def close_session(self):
        self.sess.close()

class Model_for_test:
    def __init__(self, tfconfig, chain):

        self.chain = chain

        self.imgs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image')
        self.labels = tf.placeholder(tf.float32, shape=[None, 3], name='label')
        self.ref_vector = tf.placeholder(tf.float32, [None, 1, 1, 512])
        self.test_vector = tf.placeholder(tf.float32, [None, 1, 1, 512])

        cropped_imgs = tf.image.central_crop(self.imgs, 0.875)

        self.sess = tf.Session(config=tfconfig)

        with slim.arg_scope(vgg.vgg_arg_scope()):
            self.feature, _ = vgg.vgg_16(cropped_imgs, num_classes=0, is_training=False, fc_layers=False,
                                         spatial_squeeze=False, global_pool=True, scope='vgg_16')

            if self.chain > 1:

                case = self.feature
                case = conv1x1(case, outplanes=512, activation_fn=nn_ops.relu, name='fc1_case')
                case = dropout(case, name='do1_case')
                case = conv1x1(case, outplanes=512, activation_fn=nn_ops.relu, name='fc2_case')
                case = dropout(case, name='do2_case')

                if self.chain == 2 or self.chain == 6:
                    g_logits = conv1x1(case, outplanes=2, activation_fn=None, name='fc3_gender')
                    self.g_logits = array_ops.squeeze(g_logits, [1, 2], name='SpatialSqueeze')

                if self.chain == 3 or self.chain == 6:
                    r_logits = conv1x1(case, outplanes=3, activation_fn=None, name='fc3_race')
                    self.r_logits = array_ops.squeeze(r_logits, [1, 2], name='SpatialSqueeze')

            feature_cat = tf.concat([self.ref_vector, self.test_vector], axis=3)
            feature_cat = conv1x1(feature_cat, outplanes=512, activation_fn=nn_ops.relu, name='fc1')
            feature_cat = dropout(feature_cat, name='do1')
            feature_cat = conv1x1(feature_cat, outplanes=512, activation_fn=nn_ops.relu, name='fc2')
            feature_cat = dropout(feature_cat, name='do2')
            f_logits = conv1x1(feature_cat, outplanes=3, activation_fn=None, name='fc3')
            f_logits = array_ops.squeeze(f_logits, [1, 2], name='SpatialSqueeze')
            self.f_logits_softmax = slim.softmax(f_logits)

            self.f_loss_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=f_logits, labels=self.labels))

    def load_model(self, model_path):
        print('Load Model...')

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

        print('Finish...')


    def extract_features(self, imgs_batch):
        return self.sess.run(self.feature, feed_dict={self.imgs: imgs_batch})

    def get_score(self, ref_features, test_features, labels):
        return self.sess.run(self.f_loss_softmax, feed_dict={self.ref_vector: ref_features, self.test_vector: test_features, self.labels: labels})

    def pred_gender(self, imgs_batch):
        return self.sess.run(tf.argmax(self.g_logits, 1), feed_dict={self.imgs: imgs_batch})

    def pred_race(self, imgs_batch):
        return self.sess.run(tf.argmax(self.r_logits, 1), feed_dict={self.imgs: imgs_batch})

    def pred_order(self, ref_vectors_batch, test_vectors_batch):
        return self.sess.run(tf.one_hot(tf.argmax(self.f_logits_softmax, 1), 3), feed_dict={self.ref_vector: ref_vectors_batch, self.test_vector: test_vectors_batch})

    def close_session(self):
        self.sess.close()

