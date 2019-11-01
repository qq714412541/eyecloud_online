import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms, datasets


# channel means and standard deviations of kaggle dataset
MEAN = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
STD = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]


class CropMain(object):
    def __init__(self, crop_size=448):
        self.crop_size = crop_size

    def __call__(self, img):
        blurred = img.filter(ImageFilter.BLUR)
        ba = np.array(blurred)
        h, w, _ = ba.shape

        if w > 1.2 * h:
            left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
            right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
            max_bg = np.maximum(left_max, right_max)

            foreground = (ba > max_bg + 10).astype(np.uint8)
            bbox = Image.fromarray(foreground).getbbox()

            if bbox:
                left, upper, right, lower = bbox
                if right - left < 0.8 * h or lower - upper < 0.8 * h:
                    bbox = None
        else:
            bbox = None

        cropped = img if bbox is None else img.crop(bbox)
        resized = cropped.resize([self.crop_size, self.crop_size])
        return resized

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


def get_preprocess_transform(input_size):
    return transforms.Compose([
        CropMain(input_size),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD))
    ])
