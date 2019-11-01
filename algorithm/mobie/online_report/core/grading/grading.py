import sys

import torch
from PIL import Image

from . import data_utils
sys.path.insert(0, './algorithms/mobile/online_report/core/grading')


class Grader:
    def __init__(self, config):
        self.model_path = config['model_path']
        self.input_size = config['input_size']
        self.device = config['device']

        self.thresholds = [-0.5, 0.5, 1.5, 2.5, 3.5]

    def initial(self):
        torch.set_grad_enabled(False)
        self.model = torch.load(self.model_path).eval().to(self.device)
        self.preprocess = data_utils.get_preprocess_transform(self.input_size)

    def grade(self, filepath):
        image = Image.open(filepath)
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        pred = self.model(image).item()
        print(pred)
        inferred_class, prob = self.classify(pred)

        return prob, inferred_class

    def classify(self, predict):
        thresholds = self.thresholds
        predict = min(max(predict, thresholds[0]), thresholds[-1]+1)
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                prob = 1 - (predict - i) ** 2
                return i, prob
