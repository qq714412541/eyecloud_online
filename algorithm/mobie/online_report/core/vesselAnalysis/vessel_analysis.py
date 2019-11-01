from xlwt import *
import time
import collections
import statistics
from skimage import morphology, feature
from scipy import ndimage
from scipy import signal
from skimage import measure
from skimage import morphology
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.ticker import FuncFormatter
import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VesselAnalysis:

    def __init__(self, config):
        self.config = config

    def analysis(self, FilePath_rgb, FilePath_bw, disc_spot, width):
        image_rgb = img.imread(FilePath_rgb)
        image_bw = img.imread(FilePath_bw)

        k = 3
        image_bw_up = cv2.resize(image_bw, (0, 0), fx=k, fy=k)

        img_name = FilePath_bw.split('/')[-1]
        x0, y0 = disc_spot
        srcimage = cv2.imread(FilePath_rgb)
        col = srcimage.shape[1]
        row = srcimage.shape[0]
        disc_radius = col * width / 6 * k
        cv2.circle(srcimage, (int(x0 * col), int(y0 * row)),
                   int(col * width * 2), (255, 255, 255), 5)
        cv2.circle(srcimage, (int(x0 * col), int(y0 * row)),
                   int(col * width * 6), (255, 255, 255), 5)
        if x0 < 0.5:
            cv2.putText(srcimage, 'ROI', (int(x0 * col + col * width * 2), int(y0 * row)),
                        cv2.FONT_HERSHEY_COMPLEX, 5, (255, 255, 255), 2)
        else:
            cv2.putText(srcimage, 'ROI', (int(x0 * col - col * width * 6), int(y0 * row)),
                        cv2.FONT_HERSHEY_COMPLEX, 5, (255, 255, 255), 2)

        cv2.imwrite(self.config['path_to_images_output'], srcimage)

        image_roi, density = roi(image_bw_up, x0, y0, width)
        diameter, length = parameters(image_roi, k)
        diameter = np.array(diameter) / disc_radius

        length = np.array(length) / disc_radius

        segments = len(diameter)
        counter1 = collections.Counter(diameter)
        counter_diameter = []
        for k, v in counter1.items():
            counter_diameter.append([k, v])
        counter_diameter = np.array(counter_diameter)

        counter2 = collections.Counter(length)
        counter_length = []
        for k, v in counter2.items():
            counter_length.append([k, v])
        counter_length = np.array(counter_length)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        iqa = (average_gradient(image_gray) + edge_intensity(image_gray)) / 2
        if iqa < 4:
            quality = '差'
        elif iqa < 7:
            quality = '中'
        else:
            quality = '良'

        stat = {
            'density': density,
            'num of segments': segments,
            'mean_diameter': np.mean(diameter),
            'std_diameter': np.std(diameter),
            'mean_length': np.mean(length),
            'std_length': np.std(length),
            'iqa': iqa,
            'quality': quality
        }

        merge_draw(counter_diameter, self.config['path_to_save_counter_diameter'], 0.04)
        merge_draw(counter_length,
                   self.config['path_to_save_counter_length'], 0.5)
        return (
            stat,
            self.config['path_to_images_output'],
            self.config['path_to_save_counter_diameter'],
            self.config['path_to_save_counter_length']
        )


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def roi(image_bw, x0, y0, width):
    m, n = image_bw.shape
    roi = np.array(image_bw)
    k = 0
    for i in range(m):
        for j in range(n):
            center = np.array([int(m * y0), int(n * x0)])
            t = np.array([i, j])
            if (sum((t - center) ** 2)) ** (1 / 2) < width * n * 2:
                roi[i, j] = 0
            elif (sum((t - center) ** 2)) ** (1 / 2) > width * n * 6:
                roi[i, j] = 0
            else:
                k = k + 1
    roi[roi < 0.5] = 0
    roi[roi > 0.5] = 1
    ma = np.nonzero(roi)
    density = len(ma[0]) / k
    density = "%.2f%%" % (density * 100)
    return roi, density


def parameters(image_roi, k):

    kernel = np.ones((5, 5), np.uint8)
    opening_roi = cv2.morphologyEx(image_roi, cv2.MORPH_OPEN, kernel)
    closing_roi = cv2.morphologyEx(opening_roi, cv2.MORPH_CLOSE, kernel)
    skel_roi = morphology.skeletonize(closing_roi)
    skel_roi = 1 * skel_roi

    fil = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])
    boundary_roi = signal.convolve2d(image_roi, fil, 'same')
    boundary_roi[boundary_roi > 7] = 0
    boundary_roi = boundary_roi * image_roi
    boundary_roi[boundary_roi > 0] = 1
    dis = ndimage.morphology.distance_transform_edt(boundary_roi == 0)

    res = signal.convolve2d(skel_roi, fil, 'same')
    res[res != 3] = 0
    skel_vessel = res * skel_roi
    skel_vessel[skel_vessel > 0] = 1
    label_image = measure.label(skel_vessel)
    num = np.max(label_image)
    d = []
    l = []
    for p in range(num):
        image_test = np.array(label_image)
        length = sum(sum(label_image == p))

        if length > 15 * k:
            image_test[image_test != p] = 0
            image_test[image_test == p] = 2
            dia = image_test * dis

            x = np.nonzero(dia)
            diameters0 = np.trunc(dia[x])
            counter1 = collections.Counter(diameters0)
            diameters = counter1.most_common(1)[0][0]
            if diameters > 2 * k and length < 300 * k:
                d.append(diameters)
                l.append(length)
    return d, l


def merge_draw(counter, path, n):
    data = {}
    xs = counter[:, 0]
    ys = counter[:, 1]
    ys = ys / sum(ys)
    for i in range(len(xs)):
        c = xs[i] // n * n
        if c in data.keys():
            data[c] += ys[i]
        else:
            data[c] = ys[i]

    plt.bar(
        data.keys(),
        data.values(),
        n,
        align='edge',
        edgecolor='black',
        color='xkcd:azure'
    )

    def to_percent(temp, position):
        return '%1.0f' % (100 * temp) + '%'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.xticks(np.arange(0, n * 12, n))
    plt.xlabel('ODR')
    plt.ylabel('frequency%')
    plt.savefig(path)
    plt.close()


def average_gradient(I):
    m, n = I.shape
    fil = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]])
    out1 = signal.convolve2d(I, fil, 'same')

    out2 = signal.convolve2d(I, fil.transpose(), 'same')
    f = np.sum(pow(pow(out1, 2) + pow(out2, 2), 0.5))

    avg = round(f / m / n / 0.21, 1)
    return avg


def edge_intensity(I):
    m, n = I.shape
    fil = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
    out1 = signal.convolve2d(I, fil, 'same')

    out2 = signal.convolve2d(I, fil.transpose(), 'same')
    f = np.sum(pow(pow(out1, 2) + pow(out2, 2), 0.5))
    edg = round(f / m / n / 1.42, 1)
    return edg


if __name__ == "__main__":
    test = VesselAnalysis(config)
    test.initial()
    test.analysis('./img_rgb/131.jpg', './img_bw/131.png')
