# -*- coding: utf-8 -*-
import numpy as np
import cv2

from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from .util import inception_preprocess, vessel_preprocess, preprocess, genMasks, extract_patches, paint_border, recompone_overlap

class Trainer():
    def __init__(self, model, model_name, data, config):
        self.model = model
        self.data = data
        self.config = config
        self.model_name = model_name
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.config.model_path + self.config.model_name + '_best_weights.h5',
                verbose=1,
                monitor='val_loss',
                mode='auto',
                save_best_only=True
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.checkpoint,
                write_images=True,
                write_graph=True,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                patience=self.config.early_stopping
            )
        )

    def train(self):           
        database = DataGenerator(self.config, data=self.data, model=self.model_name)
        x_train, y_train = database._train_data()
        x_val, y_val, _, _ = database._val_data()
        #_, x_val, _, y_val= train_test_split(x_val, y_val, test_size=5120)
        '''
        train_datagen = ImageDataGenerator(
                rotation_range=90,
                horizontal_flip=True,
                vertical_flip=True,
            )
        '''
        train_datagen = ImageDataGenerator()
        steps = int(self.config.sample * x_train.shape[0] / self.config.batch_size)
        val_steps = int(self.config.sample * x_val.shape[0] / self.config.batch_size)

        self.model.fit_generator(
            generator=train_datagen.flow(x_train, y_train, batch_size=self.config.batch_size),
            epochs=self.config.epochs,
            steps_per_epoch=steps,
            validation_steps=val_steps,
            verbose=1,
            callbacks=self.callbacks,
            validation_data=train_datagen.flow(x_val, y_val, batch_size=self.config.batch_size),
        )

        self.model.save_weights(self.config.model_path + self.config.model_name + '_last_weights.h5', overwrite=True)

class DataGenerator():
    """
    load image (Generator)
    """
    def __init__(self, config, model, data=None, test_data=None):
        self.config = config
        if test_data is None:
            if model == 'DeepLabv3+':
                self.train_img = inception_preprocess(data[0])
                self.train_gt = data[1] / 255.
                self.val_img = inception_preprocess(data[2])
                self.val_gt = data[3] / 255.
            elif self.config.dataset_name in ['vessel', 'vessel_aug']:
                self.train_img = vessel_preprocess(data[0])
                self.train_gt = data[1] / 255.
                self.val_img = vessel_preprocess(data[2])
                self.val_gt = data[3] / 255.
            else:
                self.train_img = preprocess(data[0])
                self.train_gt = data[1] / 255.
                self.val_img = preprocess(data[2])
                self.val_gt = data[3] / 255.
        else:
            # test time DataGenerator
            if model == 'DeepLabv3+':
                self.test_img = inception_preprocess(test_data)
            elif self.config.dataset_name == 'vessel':
                self.test_img = vessel_preprocess(test_data)
            else:
                self.test_img = preprocess(test_data)

    def center_sampling(self, image, mask):
        x_center = np.random.randint(0 + int(self.config.patch_width / 2), self.config.width - int(self.config.patch_width / 2))
        y_center = np.random.randint(0 + int(self.config.patch_height / 2), self.config.height - int(self.config.patch_height / 2))

        if y_center < self.config.patch_height / 2:
            y_center = self.config.patch_height / 2
        elif y_center > self.config.height - self.config.patch_height / 2:
            y_center = self.config.height - self.config.patch_height / 2

        if x_center < self.config.patch_width / 2:
            x_center = self.config.patch_width / 2
        elif x_center > self.config.width - self.config.patch_width / 2:
            x_center = self.config.width - self.config.patch_width / 2

        image_patch = image[int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2), int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2), :]
        mask_patch = mask[int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2), int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2), :]

        return image_patch, mask_patch

    def prepare_data(self, train_imgs, train_masks, patches_mode=True):
        train_imgs, train_masks = shuffle(train_imgs, train_masks)
        if patches_mode == False:
            Nimgs = train_imgs.shape[0]
            X = np.zeros([Nimgs, self.config.height, self.config.width, 3])
            Y = np.zeros([Nimgs, self.config.height * self.config.width, self.config.seg_num + 1])
            for i in range(Nimgs):
                X[i, :, :, :] = train_imgs[i, :, :, :]
                Y[i, :, :] = genMasks(train_masks[i, :, :, :], self.config.seg_num + 1)
        else:
            Nimgs = train_imgs.shape[0]
            X = np.zeros([Nimgs*self.config.sample, self.config.patch_height, self.config.patch_width, 1])
            Y = np.zeros([Nimgs*self.config.sample, self.config.patch_height * self.config.patch_width, self.config.seg_num + 1])
            for i in range(Nimgs):
                for j in range(self.config.sample):
                    X[i*self.config.sample+j, :, :, :], sample_mask = self.center_sampling(train_imgs[i, :, :, :], train_masks[i, :, :, :])
                    Y[i*self.config.sample+j, :, :] = genMasks(sample_mask, self.config.seg_num + 1)

        return X, Y

    def prepare_test_data(self, imgs, gts=None, patches_mode=True):
        if patches_mode == False:
            Nimgs = imgs.shape[0]
            X = np.zeros([Nimgs, self.config.height, self.config.width, 3])
            Y = np.zeros([Nimgs, self.config.height * self.config.width, self.config.seg_num + 1])
            for i in range(Nimgs):
                X[i, :, :, :] = imgs[i, :, :, :]
                Y[i, :, :] = genMasks(gts[i, :, :, :], self.config.seg_num + 1)

            return X, Y, None, None
        elif gts is None:
            test_img = paint_border(imgs, self.config)
            test_img_patches = extract_patches(test_img, self.config)
            return test_img_patches, test_img.shape[1], test_img.shape[2]
        else:
            test_imgs_patches = [] 
            test_gts_patches = []
            # Padding
            print(imgs.shape)
            for i in range(imgs.shape[0]):
                test_img = paint_border(imgs[i].reshape([1, imgs.shape[1], imgs.shape[2], imgs.shape[3]]), self.config)
                test_img_patches = extract_patches(test_img, self.config)
                test_imgs_patches.extend(test_img_patches)
                if gts is not None:
                    test_gt = paint_border(gts[i].reshape([1, imgs.shape[1], imgs.shape[2], imgs.shape[3]]), self.config)
                    test_gt_patches = extract_patches(test_gt, self.config)
                    test_mask_patches = np.empty(shape=[test_gt_patches.shape[0], test_gt_patches.shape[1] * test_gt_patches.shape[2], self.config.seg_num + 1])
                    for j in range(test_gt_patches.shape[0]):
                        test_mask_patches[j] = genMasks(test_gt_patches[j], self.config.seg_num + 1)
                    test_gts_patches.extend(test_mask_patches)

            return np.array(test_imgs_patches), np.array(test_gts_patches), test_img.shape[1], test_img.shape[2]

    def _train_data(self):
        return self.prepare_data(self.train_img, self.train_gt, patches_mode=self.config.patches_mode)

    def _val_data(self):
        return self.prepare_test_data(self.val_img, gts=self.val_gt, patches_mode=self.config.patches_mode)

    def _test_data(self):
        return self.prepare_test_data(self.test_img)