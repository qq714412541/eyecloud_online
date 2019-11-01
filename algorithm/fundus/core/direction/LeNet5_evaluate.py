import numpy as np
import tensorflow as tf
from . import LeNet5_inference
import os


class Direction:
    def __init__(self, config):
        graph = tf.Graph()
        with graph.as_default() as g:
            self.x = tf.placeholder(tf.float32, [
                1,
                LeNet5_inference.IMAGE_SIZE,
                LeNet5_inference.IMAGE_SIZE,
                LeNet5_inference.NUM_CHANNELS],
                name='x-input')

            self.y = LeNet5_inference.inference(self.x, False, None)

            # variable_averages = tf.train.ExponentialMovingAverage(0.99)
            # variables_to_restore = variable_averages.variables_to_restore()
            # saver = tf.train.Saver(variables_to_restore)
            saver = tf.train.Saver()

        sess = tf.Session(graph=graph)
        ckpt = tf.train.get_checkpoint_state(config['model_path'])
        saver.restore(sess, ckpt.model_checkpoint_path)

        self.graph = graph
        self.sess = sess

    def evalutate(self, image):
        with self.graph.as_default():
            with self.sess.as_default():
                image_raw_data = tf.gfile.FastGFile(image, 'rb').read()
                img = tf.image.decode_jpeg(image_raw_data)
                resized_img = tf.image.resize_images(img, (32, 32), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                resized_img = np.asarray(resized_img.eval(session=self.sess), dtype='uint8')
                image_x = resized_img.astype('float32') / 255.0
                image_x = np.array(image_x).reshape(1, 32, 32, 3)
                res = self.sess.run(self.y, feed_dict={self.x: image_x})

                if res[0][0] > res[0][1]:
                    return '左眼', str(round(res[0][0], 2))
                else:
                    return '右眼', str(round(res[0][1], 2))
