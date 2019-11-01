# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random

from .util import vessel_preprocess, preprocess, genMasks, extract_patches, paint_border, recompone_overlap

class DataGenerator():
    """
    load image (Generator)
    """
    def __init__(self, config, data=None, test_data=None):
        self.config = config
        if test_data is None:
            self.train_img = preprocess(data[0])
            self.train_gt = data[1] / 255.
            self.mask = data[2]
            self.val_img = preprocess(data[3])
            self.val_gt = preprocess(data[4])
        else:
            # test time DataGenerator
            self.test_img = preprocess(test_data)

    def center_sampling(self, image, mask, fov):
        x_center = 0
        y_center = 0
        while fov[y_center, x_center] != 0:
          x_center = np.random.randint(0 + int(self.config.patch_width/2), self.config.width - int(self.config.patch_width/2))
          y_center = np.random.randint(0 + int(self.config.patch_height/2), self.config.height - int(self.config.patch_height/2))

        if y_center < self.config.patch_height/2:
            y_center = self.config.patch_height/2
        elif y_center > self.config.height - self.config.patch_height/2:
            y_center = self.config.height - self.config.patch_height/2

        if x_center < self.config.patch_width/2:
            x_center = self.config.patch_width/2
        elif x_center > self.config.width - self.config.patch_width/2:
            x_center = self.config.width - self.config.patch_width/2

        image_patch = image[int(y_center - self.config.patch_height/2):int(y_center + self.config.patch_height/2), int(x_center - self.config.patch_width/2):int(x_center + self.config.patch_width/2),:]
        mask_patch = mask[int(y_center - self.config.patch_height/2):int(y_center + self.config.patch_height/2), int(x_center - self.config.patch_width/2):int(x_center + self.config.patch_width/2),:]

        return image_patch, mask_patch

    def prepare_data(self, train_imgs, train_masks, masks, patches_mode=True):
        train_imgs, train_masks = shuffle(train_imgs, train_masks)
        if patches_mode == False:
            Nimgs = train_imgs.shape[0]
            X = np.zeros([Nimgs, self.config.height, self.config.width, 3])
            Y = np.zeros([Nimgs, self.config.height, self.config.width, self.config.seg_num+1])
            for i in range(Nimgs):
                X[i,:,:,:] = train_imgs[i,:,:,:]
                Y[i,:,:,:] = genMasks(train_masks[i,:,:,:], self.config.seg_num+1)
        else:
            Nimgs = train_imgs.shape[0]
            X = np.zeros([Nimgs*self.config.sample, self.config.patch_height, self.config.patch_width, 3])
            Y = np.zeros([Nimgs*self.config.sample, self.config.patch_height, self.config.patch_width, self.config.seg_num+1])
            for i in range(Nimgs):
                for j in range(self.config.sample):
                    X[i*self.config.sample+j,:,:,:], sample_mask = self.center_sampling(train_imgs[i,:,:,:], train_masks[i,:,:,:], masks[i,:,:])
                    Y[i*self.config.sample+j,:,:,:] = genMasks(sample_mask, self.config.seg_num+1)

        #return X, Y
        return X, np.expand_dims(Y[:,:,:,1], axis=-1)

    def prepare_test_data(self, imgs, gts=None):
        if gts is None:
            test_img = paint_border(imgs, self.config)
            test_img_patches = extract_patches(test_img, self.config)
            return test_img_patches, test_img.shape[1], test_img.shape[2]
        else:
            test_imgs_patches = [] 
            test_gts_patches = []
            # Padding
            print(imgs.shape)
            for i in range(imgs.shape[0]):
                # cornea mask
                #for x in range(imgs.shape[1]):
                #    for y in range(imgs.shape[2]):
                #        if self.mask[i,x,y] != 0:
                #            imgs[i,x,y] = np.asarray([0,0,0])
                # to patches
                test_img = paint_border(imgs[i].reshape([1, imgs.shape[1], imgs.shape[2], imgs.shape[3]]), self.config)
                test_img_patches = extract_patches(test_img, self.config)
                test_imgs_patches.extend(test_img_patches)
                test_gt = paint_border(gts[i].reshape([1, imgs.shape[1], imgs.shape[2], gts[i].shape[-1]]), self.config)
                test_gt_patches = extract_patches(test_gt, self.config)
                test_mask_patches = np.empty(shape=[test_gt_patches.shape[0], test_gt_patches.shape[1], test_gt_patches.shape[2], self.config.seg_num+1])
                for j in range(test_gt_patches.shape[0]):
                    test_mask_patches[j] = genMasks(test_gt_patches[j], self.config.seg_num+1)
                test_gts_patches.extend(test_mask_patches)

            #return np.array(test_imgs_patches), np.array(test_gts_patches), test_img.shape[1], test_img.shape[2]
            return np.array(test_imgs_patches), np.expand_dims(np.array(test_gts_patches)[:,:,:,1], axis=-1)

    def _train_data(self):
        return self.prepare_data(self.train_img, self.train_gt, self.mask, patches_mode=self.config.patches_mode)

    def _val_data(self):
        return self.prepare_test_data(self.val_img, gts=self.val_gt)

    def _test_data(self):
        return self.prepare_test_data(self.test_img)