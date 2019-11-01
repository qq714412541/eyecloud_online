import tensorflow as tf
import numpy as np
import cv2 as cv

from .model import StageModel
from .data_utils import DataManager, generate_lesion_labels, stack
from .retinanet.keras_retinanet.utils.image import read_image_bgr, resize_image
from .configs import network_configs
from .configs import FLAGS


class Inference:
    def __init__(self, config):
        self.model_path = config['model_path']
        self.num_classes = config['num_classes']
        self.lesion_num = config['lesion_num']
        self.FLAGS = config['FLAGS']
        self.label2class = config['label2class']

        self.config = network_configs[config['network']]

    def initial(self):
        for key in self.FLAGS.keys():
            FLAGS[key] = self.FLAGS[key]

        graph = tf.Graph()
        with graph.as_default():
            model = StageModel()
            model.build(None, self.num_classes)
            saver = tf.train.Saver()

        # restore
        sess = tf.Session(graph=graph)
        saver.restore(sess, self.model_path)

        self.model = model
        self.graph = graph
        self.sess = sess

    def classify(self, filepath, lesion_result):
        with self.graph.as_default():
            with self.sess.as_default():
                config = self.config
                model = self.model

                # stack image with lesion labels
                image = read_image_bgr(filepath)
                image, _ = resize_image(image)

                # generate lesion label
                lesion_labels = generate_lesion_labels(
                    lesion_result,
                    image.shape[:2],
                    self.lesion_num
                )
                stacked_image = stack(config, image, lesion_labels, self.lesion_num)

                feed_dict = {
                    model.is_training: False,
                    model.input: [stacked_image]
                }
                softmax, inferred_class = self.sess.run(
                    [model.softmax, model.inferred_class],
                    feed_dict=feed_dict
                )
                print(softmax, inferred_class)
                return np.max(softmax[0]), self.label2class[str(inferred_class[0])]


# test
if __name__ == "__main__":
    import os
    import json
    from retinanet.keras_retinanet import models
    from tqdm import tqdm
    config = json.load(open('./config.json', 'r'))['dr']
    images = os.listdir('../../../dataset/new_train_data/new_train_data/test/DR')
    drdetector = Inference(config)
    drdetector.initial()
    lesion_model = models.load_model('../../models/lesion/combined50.h5', backbone_name='resnet50')

    test = [0, 0]
    for image in tqdm(images):
        image_path = os.path.join('../../../dataset/new_train_data/new_train_data/test/DR', image)
        image = read_image_bgr(image_path)
        image, _ = resize_image(image)
        # image = preprocess_image(image, mode='caffe')
        image_feed = np.expand_dims(image, axis=0)
        lesion_result = lesion_model.predict_on_batch(image_feed)
        test[drdetector.classify(image_path, lesion_result)[1]] += 1
    print(test)
