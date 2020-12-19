import os
import numpy as np
from random import random
from random import choice
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
    
def data_augment(img, opt):
    img = np.array(img)
    img = random_crop(img)       # applying random crop on each image, cropping it to 224 by 224
    img = random_hflip(img)      # applying random horizontal flip with the probability of 0.5, as mentioned in the paper
    
    if random() < opt['blur_prob']:                       # Applying random blurs with th probability of opt['blur_prob']
        sigma = continuous_sampling(opt['blur_sig'])
        gaussian_blur(img, sigma)

    if random() < opt['jpg_prob']:                       # Applying random jpeg compression with the probability of opt['jpg_prob']
        method = discrete_sampling(opt['jpg_method'])
        qf = discrete_sampling(opt['jpg_qual'])
        img = jpeg_from_key(img, qf, method)

    return img

def random_hflip(img, prob=0.5):
    
    if random() < 0.5:
        img = cv2.flip(img,1)
     
    return img

def random_crop(img, size=224):
    h, w = img.shape[:2]
    y = np.random.randint(0, h-size)
    x = np.random.randint(0, w-size)
    img = img[y:y+size, x:x+size, :]

    return img

def continuous_sampling(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def discrete_sampling(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg_comp(img, compress_factor):
    img_cv2 = img[:,:,::-1]
    quality_factor = [int(cv2.IMWRITE_JPEG_QUALITY), compress_factor]
    result, encimg = cv2.imencode('.jpg', img_cv2, quality_factor)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg_comp(img, compress_factor):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_factor)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg_comp, 'pil': pil_jpg_comp}
def jpeg_from_key(img, compress_factor, key):
    method = jpeg_dict[key]
    return method(img, compress_factor)
