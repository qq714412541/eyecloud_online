# import keras
import cv2 as cv
import time
import numpy as np
import os
import cv2
import keras

# import keras_retinanet
from .retinanet.keras_retinanet import models
from .retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from .retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from .retinanet.keras_retinanet.utils.colors import label_color

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


class Lesion:
    def __init__(self, config):
        self.config = config

    def initial(self):
        self.model = models.load_model(
            self.config['model_path'], backbone_name='resnet50')

    def lesion_locate(self, image_path, spot, disc_radius):
        f_name = os.path.split(image_path)[1]
        exu_name = 'exu_' + f_name
        exu_output_path = os.path.join(self.config['output_dir'], exu_name)
        hem_name = 'hem_' + f_name
        hem_output_path = os.path.join(self.config['output_dir'], hem_name)

        image = read_image_bgr(image_path)
        draw_exu = image.copy()
        draw_hem = image.copy()

        # preprocess image for network
        image, scale = resize_image(image)
        image = preprocess_image(image, mode='caffe')
        image_feed = np.expand_dims(image, axis=0)

        lesion_result = self.model.predict_on_batch(image_feed)
        boxes, scores, labels = (item.copy() for item in lesion_result)
        boxes /= scale

        color = self.color2num(self.config['color'])
        exu_distances = []
        hem_distances = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < self.config['threshold']:
                break

            b = box.astype(int)
            distance = self.get_distance(b, spot, disc_radius)
            if label == self.config['prefix']['exu']:
                exu_distances.append(distance)
                draw_box(draw_exu, b, color=[255, 30, 30])
            elif label == self.config['prefix']['hem']:
                hem_distances.append(distance)
                draw_box(draw_hem, b, color=[0, 255, 127])

        def generate_result(prefix, distances, output_path, draw):
            name = prefix + '_plot' + f_name
            plot_path = os.path.join(self.config['plot_dir'], name)
            lesion_nums = len(distances)
            mean_distance = 0 if lesion_nums == 0 or disc_radius <= 0 else sum(distances) / lesion_nums
            self.plot(draw.shape, distances, plot_path)
            cv.imwrite(output_path, draw)
            return lesion_nums, mean_distance, plot_path, output_path

        re_exu = generate_result('exu', exu_distances, exu_output_path, draw_exu)
        re_hem = generate_result('hem', hem_distances, hem_output_path, draw_hem)

        return (re_exu, re_hem, lesion_result)

    def get_distance(self, box, spot, disc_radius):
        x1, y1, x2, y2 = box
        loc = ((x1+x2)/2, (y1+y2)/2)
        distance = ((loc[0]-spot[0])**2 + (loc[1]-spot[1])**2)**0.5
        distance /= disc_radius
        return distance

    def plot(self, shape, distances, output_path):
        dia_distance = (shape[0]**2 + shape[0]**2)**0.5 // 187
        # min_distance = min(distances)
        # max_distance = max(distances)
        # freq = {i: 0 for i in np.arange(min_distance//0.5*0.5,
        #                                 max_distance//0.5*0.5+1,
        #                                 0.5)}
        freq = {i: 0 for i in np.arange(0, dia_distance, 0.5)}
        for distance in distances:
            # unexpected Nan problem
            try:
                freq[distance//0.5*0.5] += 1
            except:
                pass

        # alter color
        # colors = ['xkcd:azure' if i %
        #           1 == 0 else 'xkcd:lightblue' for i in freq.keys()]

        num = sum(freq.values())
        for key in freq.keys():
            freq[key] = 0 if num == 0 else freq[key] / num
        plt.bar(
            freq.keys(),
            freq.values(),
            0.5,
            align='edge',
            edgecolor='black',
            color='xkcd:azure'
        )

        def to_percent(temp, position):
            return '%1.0f' % (100*temp) + '%'
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

        plt.xticks(np.arange(0, dia_distance))
        plt.xlabel('ODR')
        plt.ylabel('frequency%')
        plt.savefig(output_path)
        plt.close()

    def color2num(self, color):
        if color == 'red':
            return (255, 0, 0)
        elif color == 'blue':
            return (0, 255, 0)
        elif color == 'green':
            return (0, 0, 255)
