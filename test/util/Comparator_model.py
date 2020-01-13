from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import util.vgg as vgg
import tensorflow.contrib.slim as slim
import tensorflow as tf

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
tf.reset_default_graph()

def conv1x1(x, outplanes, activation_fn, name):
    with arg_scope([layers.conv2d], trainable=False):
        x = layers.conv2d(x, outplanes, [1, 1], activation_fn=activation_fn, scope=name)
    return x

def dropout(x, name, prob = 0.8):
    with arg_scope([layers.dropout], is_training=False):
        x = layers.dropout(x, prob, scope=name)
    return x

class Model:
    def __init__(self, chain):

        self.chain = chain

        self.imgs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image')
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

    def load_model(self, model_path):
        print('Load Model...')

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

        print('Finish...')


    def extract_features(self, imgs_batch):
        return self.sess.run(self.feature, feed_dict={self.imgs: imgs_batch})

    def pred_gender(self, imgs_batch):
        return self.sess.run(tf.argmax(self.g_logits, 1), feed_dict={self.imgs: imgs_batch})

    def pred_race(self, imgs_batch):
        return self.sess.run(tf.argmax(self.r_logits, 1), feed_dict={self.imgs: imgs_batch})

    def pred_order(self, ref_vectors_batch, test_vectors_batch):
        return self.sess.run(tf.one_hot(tf.argmax(self.f_logits_softmax, 1), 3), feed_dict={self.ref_vector: ref_vectors_batch, self.test_vector: test_vectors_batch})

    def close_session(self):
        self.sess.close()

