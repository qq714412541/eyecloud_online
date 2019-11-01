# -*- coding: utf-8 -*-
import numpy as np
import cv2
from keras.models import model_from_json
from matplotlib.pyplot import imread




class Inference:
    def __init__(self, config):
        self.model_path = config['model_path']
        self.std = config['std']
        self.mean = config['mean']
        self.image_size = config['image_size']



    def initial(self):
        self.model = model_from_json(
            open(self.model_path + 'res_architecture.json').read())
        self.model.load_weights(self.model_path + 'res_best_weights.h5', by_name=True)


    def _process(self, imgs):
        if len(imgs.shape) == 3:
            imgs = np.expand_dims(imgs, axis=0)  # turn into 4D arrays
        imgs_normalized = np.empty(imgs.shape)
        imgs_std = self.std
        imgs_mean = self.mean
        for i in range(imgs.shape[0]):
            imgs_normalized[i] = (imgs[i] - imgs_mean) / imgs_std
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                        np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))
        return imgs_normalized

    def classify(self, filepath):
        image = imread(filepath)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = self._process(image)
        score = int(self.model.predict(image)[0][0] * 100)
        if score <= 30:
            grade = '差'
        elif score > 30 and score < 80:
            grade = '中'
        else:
            grade = '好'
        return score, grade




# test
if __name__ == "__main__":
    import json
    config = json.load(open('./isa_config.json', 'r'))
    iqa_detector = Inference(config)
    iqa_detector.initial()
    filepath = './test4.jpeg'
    score = iqa_detector.classify(filepath)
    print(score)
