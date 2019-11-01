# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2
from keras.models import model_from_json
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import scipy
from skimage.measure import label, regionprops
import math


class Inference:
    def __init__(self, config):
        self.config = config

    def initial(self):
        self.detector = model_from_json(
            open(self.config['detect_model_path'] + 'detector_architecture.json').read())
        #self.detector = self._test()
        self.detector.load_weights(self.config['detect_model_path'] + 'weights.h5', by_name=True)
        self.model = model_from_json(
            open(self.config['segment_model_path'] + 'segmentor_architecture.json').read())
        self.model.load_weights(self.config['segment_model_path'] + 'weights.h5', by_name=True)

    def _process(self, imgs):
        if len(imgs.shape) == 3:
            imgs = np.expand_dims(imgs, axis=0)  # turn into 4D arrays
        imgs_normalized = np.empty(imgs.shape)
        imgs_std = self.config['std']
        imgs_mean = self.config['mean']
        for i in range(imgs.shape[0]):
            imgs_normalized[i] = (imgs[i] - imgs_mean) / imgs_std
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))
        return imgs_normalized

    def BW_img(self, input, thresholding):
        if input.max() > thresholding:
            binary = input > thresholding
        else:
            binary = input > input.max() / 2.0

        label_image = label(binary)
        regions = regionprops(label_image)
        area_list = []
        for region in regions:
            area_list.append(region.area)
        if area_list:
            idx_max = np.argmax(area_list)
            binary[label_image != idx_max + 1] = 0
        return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

    def extract_optic(self, org_img, DiscROI_size, C_x, C_y):
        c = np.size(org_img, 2)
        tmp_size = int(DiscROI_size / 2)
        disc_region = np.zeros((DiscROI_size, DiscROI_size, c), dtype=org_img.dtype)
        crop_coord = np.array([C_x - tmp_size, C_x + tmp_size, C_y - tmp_size, C_y + tmp_size], dtype=int)
        err_coord = [0, DiscROI_size, 0, DiscROI_size]

        if crop_coord[0] < 0:
            err_coord[0] = abs(crop_coord[0])
            crop_coord[0] = 0

        if crop_coord[2] < 0:
            err_coord[2] = abs(crop_coord[2])
            crop_coord[2] = 0

        if crop_coord[1] > org_img.shape[0]:
            err_coord[1] = err_coord[1] - (crop_coord[1] - org_img.shape[0])
            crop_coord[1] = org_img.shape[0]

        if crop_coord[3] > org_img.shape[1]:
            err_coord[3] = err_coord[3] - (crop_coord[3] - org_img.shape[1])
            crop_coord[3] = org_img.shape[1]

        disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = org_img[crop_coord[0]:crop_coord[1],
                                                                                      crop_coord[2]:crop_coord[3], ]

        return disc_region, err_coord, crop_coord

    def _detect(self, ori_img):
        ori_shape = np.shape(ori_img)
        img = cv2.resize(ori_img, (self.config['disc_size'], self.config['disc_size']))
        img = np.expand_dims(img, axis=0)
        disc_map = self.detector.predict(img)
        disc_map = self.BW_img(np.reshape(disc_map, (self.config['disc_size'], self.config['disc_size'])), 0.5)
        regions = regionprops(label(disc_map))
        C_x = int(regions[0].centroid[0] * ori_shape[0] / self.config['disc_size'])
        C_y = int(regions[0].centroid[1] * ori_shape[1] / self.config['disc_size'])
        mini_img, err_coord, crop_coord = self.extract_optic(ori_img, self.config['ROI_size'], C_x, C_y)
        return mini_img

    def _segment(self, img):
        img = self._process(img)
        segmentation = self.model.predict(img)
        return segmentation

    def ellipse_fit(self, img):
        _, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            for i in range(len(contours)):
                if len(contours[i]) >= 5:
                    ellipse = cv2.fitEllipse(contours[i])
                    if ellipse[1][0] != 0 and ellipse[1][1] != 0:
                        return ellipse


    def run(self, filepath):
        image = imread(filepath)
        disc_image = self._detect(image)
        disc_image = cv2.resize(disc_image, (self.config['segment_size'], self.config['segment_size']))
        segmentation = (self._segment(disc_image) >= 0.5).astype(np.uint8)
        segmentation = np.squeeze(segmentation, axis=0)
        try:
            disc_ellipse = self.ellipse_fit(segmentation[:, :, 1])
            cup_ellipse = self.ellipse_fit(segmentation[:, :, 2])
            cv2.ellipse(disc_image, disc_ellipse, (0, 255, 0), 1)
            cv2.ellipse(disc_image, cup_ellipse, (0, 0, 255), 1)
            cdr = (cup_ellipse[1][0] / disc_ellipse[1][0] + cup_ellipse[1][1] / disc_ellipse[1][1]) / 2
            disc_area = disc_ellipse[1][0] * disc_ellipse[1][1] * math.pi
            cup_area = cup_ellipse[1][0] * cup_ellipse[1][1] * math.pi
            fig_name = os.path.join(self.config['save_path'], 'cup_disc.jpg')
            plt.imsave(fig_name, disc_image)
            return fig_name, cdr, '{:.2f}'.format(disc_area), '{:.2f}'.format(cup_area)
        except e:
            raise RuntimeError('Cannot find Optic segmentation in this image')


# test
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    import json
    config = json.load(open('./config.json', 'r'))
    iqa_detector = Inference(config)
    iqa_detector.initial()
    filepath = './test3.jpeg'
    image, cdr, da, ca = iqa_detector.run(filepath)
    print('CDR is ', cdr)
    print(da)
    print(ca)
    plt.imshow(image)
    plt.show()
    plt.imsave('result.png', image)
