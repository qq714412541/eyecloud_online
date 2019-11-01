# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from keras.applications import imagenet_utils
from skimage import morphology, measure
import math

def mkdir_if_not_exist(dir_name, is_delete=False):
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(u'[INFO] Dir "%s" exists, deleting.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(u'[INFO] Dir "%s" not exists, creating.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False

# normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    #assert (imgs.shape[3] == 3)  # check the channel is 3
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = (imgs[i] - imgs_mean) / imgs_std
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))
    return imgs_normalized

# Contrast Limited Adaptive Histogram Equalization
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,:,:,0] = clahe.apply(np.array(imgs[i,:,:,:], dtype = np.uint8))
    return imgs_equalized

# Gamma Correction
def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    imgs_corrected = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_corrected[i,:,:,0] = cv2.LUT(np.array(imgs[i,:,:,:], dtype=np.uint8), table)
    return imgs_corrected

# Histograms Equalization
def histo_equalized(imgs):
    assert (len(data.shape) == 4)
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,:,:,:] = cv2.equalizeHist(np.array(imgs[i,:,:,:], dtype = np.uint8))
    return imgs_equalized

def preprocess(data):
    assert (len(data.shape) == 4)
    #assert (data.shape[3] == 3) # channel last
    train_imgs = dataset_normalized(data)

    return train_imgs

def vessel_preprocess(data):
    assert (len(data.shape) == 4)
    normalized_imgs = dataset_normalized(data)
    '''
    clahe_imgs = clahe_equalized(normalized_imgs)
    gamma_imgs = adjust_gamma(clahe_imgs, gamma=1.2)
    train_imgs = gamma_imgs / 255.
    '''
    return normalized_imgs

def inception_preprocess(data):
    return imagenet_utils.preprocess_input(data, mode='tf')

def connectTable(image,min_size,connect):
    label_image = measure.label(image)
    dst = morphology.remove_small_objects(label_image, min_size=min_size, connectivity=connect)
    return dst, measure.regionprops(dst)

def countWhite(image): 
    return np.count_nonzero(image)

def genMasks(masks, channels):
    """
    reverse mask for groundtruth 
    """
    assert (len(masks.shape) == 3)
    masks_op = 1 - masks
    new_masks = np.concatenate((masks, masks_op), axis=2)

    return new_masks

def gray2binary(image, threshold=0.5):
    image = (image >= threshold) * 1
    return image

def paint_border(imgs, config):
    assert (len(imgs.shape) == 4)
    img_h = imgs.shape[1]  # height of the full image
    img_w = imgs.shape[2]  # width of the full image
    leftover_h = (img_h - config.patch_height) % config.stride_height  # leftover on the h dim
    leftover_w = (img_w - config.patch_width) % config.stride_width  # leftover on the w dim
    full_imgs=None
    if (leftover_h != 0):  #change dimension of img_h
        tmp_imgs = np.zeros((imgs.shape[0],img_h+(config.stride_height-leftover_h),img_w,imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0],0:img_h,0:img_w,0:imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    else:
        full_imgs = imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_w+(config.stride_width - leftover_w),full_imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0],0:imgs.shape[1],0:img_w,0:full_imgs.shape[3]] =imgs
        full_imgs = tmp_imgs
    print("new full images shape: \n" +str(full_imgs.shape))
    return full_imgs

def extract_patches(full_imgs, config):
    assert (len(full_imgs.shape)==4)  #4D arrays
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image

    assert ((img_h-config.patch_height)%config.stride_height==0 and (img_w-config.patch_width)%config.stride_width==0)
    N_patches_img = ((img_h-config.patch_height)//config.stride_height+1)*((img_w-config.patch_width)//config.stride_width+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]

    patches = np.empty((N_patches_tot,config.patch_height,config.patch_width,full_imgs.shape[3]))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-config.patch_height)//config.stride_height+1):
            for w in range((img_w-config.patch_width)//config.stride_width+1):
                patch = full_imgs[i,h*config.stride_height:(h*config.stride_height)+config.patch_height,w*config.stride_width:(w*config.stride_width)+config.patch_width,:]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches

def pred_to_patches(pred, config):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)

    pred_images = np.empty((pred.shape[0],pred.shape[1],config.seg_num+1))  #(Npatches,height*width)
    pred_images[:,:,0:config.seg_num+1]=pred[:,:,0:config.seg_num+1]
    pred_images = np.reshape(pred_images,(pred_images.shape[0],config.patch_height,config.patch_width,config.seg_num+1))
    return pred_images

def recompone_overlap(preds, config, img_h, img_w):
    assert (len(preds.shape)==4)  #4D arrays

    patch_h = config.patch_height
    patch_w = config.patch_width
    N_patches_h = (img_h-patch_h)//config.stride_height+1
    N_patches_w = (img_w-patch_w)//config.stride_width+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " +str(N_patches_h))
    print("N_patches_w: " +str(N_patches_w))
    print("N_patches_img: " +str(N_patches_img))
    #assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//config.stride_height+1):
            for w in range((img_w-patch_w)//config.stride_width+1):
                full_prob[i,h*config.stride_height:(h*config.stride_height)+patch_h,w*config.stride_width:(w*config.stride_width)+patch_w,:]+=preds[k]
                full_sum[i,h*config.stride_height:(h*config.stride_height)+patch_h,w*config.stride_width:(w*config.stride_width)+patch_w,:]+=1
                k+=1
    print(k,preds.shape[0])
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    print('using avg')
    return final_avg
