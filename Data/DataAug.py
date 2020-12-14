import math
import os
import numpy as np
from random import random
from random import choice
import cv2
from PIL import Image
from tensorflow.keras.utils import Sequence
from scipy.ndimage import gaussian_filter
    
def data_augment(img, opt):
    img = np.array(img)
    img = random_crop(img)
    img = random_hflip(img)
    
    if random() < opt['blur_prob']:
        sig = sample_continuous(opt['blur_sig'])
        gaussian_blur(img, sig)

    if random() < opt['jpg_prob']:
        method = sample_discrete(opt['jpg_method'])
        qual = sample_discrete(opt['jpg_qual'])
        img = jpeg_from_key(img, qual, method)

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

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}