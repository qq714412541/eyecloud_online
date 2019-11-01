# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from tqdm import tqdm

import cv2 as cv


def main(data_dir, save_dir, model_path):
    model = models.load_model(model_path, backbone_name='resnet50')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for image in tqdm(os.listdir(data_dir)):
        image_path = os.path.join(data_dir, image)
        save_path = os.path.join(save_dir, image)
        image_name = os.path.splitext(image)[0]
        csv_path = os.path.join(save_dir, image_name + '.json')
        label(model, image_path, save_path, csv_path, label_name=False)


def label(model, image_path, save_path, csv_path, label_name=True):
    # copy to draw on
    image = read_image_bgr(image_path)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale

    # visualize detections
    with open(csv_path, 'w') as csv_file:
        csv_file.write('x1,y1,x2,y2')
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

            color = label_color(label)

            b = box.astype(int)
            csv_file.write('{},{},{},{}'.format(*b))
            draw_box(draw, b, color=color)

            caption = "{:.3f}".format(score)
            draw_caption(draw, b, caption)

    cv.imwrite(save_path, cv.cvtColor(draw, cv.COLOR_RGB2BGR))


if __name__ == "__main__":
    main('test', 'save', '../../models/hemorrhage.h5')
