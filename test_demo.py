import tensorflow.keras as keras
import tensorflow as tf
from Data.DataPipe import TestDataGenerator
from Network.Det_RN50 import Det_RN50
import pandas as pd
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default='../trained_model/model1/baseline-cp-8.ckpt')
parser.add_argument("--image",type=str,default='./demo_images/fake.png')
parser.add_argument("--crop",type=int,default=0)
args = parser.parse_args()


model = Det_RN50()
model.load_weights(args.model)
print('\n\nModel Loaded:{}'.format(args.model))
print('\nTesting on:{}\n'.format(args.image))
img = cv2.imread(args.image)
img = img/255.0
h,w,c = img.shape
img = img.reshape(1,h,w,3)

if args.crop:
    size=224
    y = int(h-size/2)
    x = int(w-size/2)
    img = img[:,y:y+size, x:x+size, :]

pred = model.predict(img)[0][0]
pred = tf.sigmoid(pred).numpy()

if pred > 0.5 :
    print('test result: 1 [image is fake]')
else:
    print('test result: 0 [image is real]')