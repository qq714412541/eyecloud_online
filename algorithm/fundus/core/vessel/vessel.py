# -*- coding: utf-8 -*-

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.callbacks import Callback
from keras.layers.core import Layer, InputSpec
from keras.utils import conv_utils
from .util import gray2binary, inception_preprocess, preprocess, pred_to_patches, recompone_overlap
from .trainer import DataGenerator
from keras import backend as K
from PIL import Image
from bunch import Bunch
import tensorflow as tf

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Vessel():
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.config = Bunch(config_dict)
        self.load_model()

    def load_model(self):
        self.model = model_from_json(open(self.config.model_architecture_path).read(), custom_objects={'BilinearUpsampling': BilinearUpsampling})
        self.model.load_weights(self.config.model_weight_path, by_name=True)

    def analyze_name(self, path):
        # Depend on system
        name = os.path.split(path)[1]
        name = os.path.splitext(name)[0]
        return name

    def predict(self, path):
        orgImg_temp = plt.imread(path)
        orgImg_temp = cv2.resize(orgImg_temp, (orgImg_temp.shape[1]//6, orgImg_temp.shape[0]//6), interpolation=cv2.INTER_AREA)
        orgImg_p = np.asarray(orgImg_temp[:,:,1]*0.75 + orgImg_temp[:,:,0]*0.25)
        orgImg = np.reshape(orgImg_p, (1, orgImg_p.shape[0], orgImg_p.shape[1], 1))
        datagen = DataGenerator(config=self.config, test_data=orgImg, model=self.config.model_name)
        test_img_patches, new_height, new_width = datagen._test_data()
        predictions = self.model.predict(test_img_patches, verbose=1)

        pred_imgs = recompone_overlap(predictions, self.config, new_height, new_width)
        pred_imgs = pred_imgs[:,0:orgImg_p.shape[0],0:orgImg_p.shape[1],:]

        probResult = pred_imgs[0,:,:,0]
        binaryResult = np.round(probResult)

        save_path = self.config_dict['save_path']+self.analyze_name(path)+".png"
        cv2.imwrite(save_path,(binaryResult*255).astype(np.uint8))
        plt.close()
        return save_path

def main(): 
    config = dict()
    config['model_architecture_path'] = './model/bnUNet_architecture.json'
    config['model_weight_path'] = './model/bnUNet_best_weights.h5'
    config['model_name'] = 'bnUNet'
    config['dataset_name'] = 'vessel'
    config['save_path'] = './result/'
    config['patch_height'] = 64
    config['patch_width'] = 64
    config['stride_height'] = 15
    config['stride_width'] = 15
    config['seg_num'] = 1
    infer = Vessel(config)
    infer.predict('./sample/left.234.jpg')

if __name__ == '__main__':
    main()
