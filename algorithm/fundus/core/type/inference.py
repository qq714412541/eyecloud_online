import tensorflow as tf
import numpy as np
import cv2 as cv

from .model import StageModel
from .configs import network_configs
from .configs import FLAGS


class Inference:
    def __init__(self, config):
        self.model_path = config['model_path']
        self.num_classes = config['num_classes']
        self.FLAGS = config['FLAGS']
        self.label2class = config['label2class']

        self.config = network_configs[config['network']]

    def initial(self):
        for key in self.FLAGS.keys():
            FLAGS[key] = self.FLAGS[key]

        graph = tf.Graph()
        with graph.as_default():
            model = StageModel()
            model.build(self.num_classes)
            saver = tf.train.Saver()

        # restore
        sess = tf.Session(graph=graph)
        saver.restore(sess, self.model_path)

        self.model = model
        self.graph = graph
        self.sess = sess

    def classify(self, filepath):
        with self.graph.as_default():
            with self.sess.as_default():
                config = self.config
                model = self.model

                image = cv.imread(filepath)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = cv.resize(
                    image,
                    (config['INPUT_WIDTH'], config['INPUT_WIDTH']),
                    interpolation=cv.INTER_LINEAR
                )
                image = image.astype(np.float32)
                image -= config['INPUT_MEAN']
                image /= config['INPUT_STD']

                feed_dict = {
                    model.is_training: False,
                    model.input: [image]
                }
                softmax, inferred_class = self.sess.run(
                    [model.softmax, model.inferred_class],
                    feed_dict=feed_dict
                )
                return np.max(softmax[0]), self.label2class[str(inferred_class[0])]


# test
if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm
    config = json.load(open('./config.json', 'r'))['type']
    images = os.listdir(r'E:\Mine\Photo\overwatch')
    type_detector = Inference(config)
    type_detector.initial()

    test = [0, 0]
    for image in tqdm(images):
        image_path = os.path.join(r'E:\Mine\Photo\overwatch', image)
        result = type_detector.classify(image_path)
        print(result)
        test[result[1]] += 1
    print(test)
