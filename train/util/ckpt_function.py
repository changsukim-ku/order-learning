import os
import pickle
import glob
import scipy.io
import cv2
import numpy as np
import tensorflow as tf
import random
from tensorflow.python import pywrap_tensorflow
from PIL import Image

def get_variables_in_checkpoint(checkpoint_path):
    try:
        print(checkpoint_path)
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed with SNAPPY.")

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []
    for v in variables:
        a = v.name.split(':')[0]
        if v.name.split(':')[0] in var_keep_dic:
            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
        else:
            print('Variables not_restored: %s' % v.name)
    return variables_to_restore

