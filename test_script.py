import tensorflow.keras as keras
import tensorflow as tf
from Data.DataPipe import TestDataGenerator
from Network.Det_RN50 import Det_RN50
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=str,default='model1/baseline-cp-8.ckpt')
parser.add_argument("--img_index",type=str,default='../CNNDetection-TF2.0/Img_index/test/seeingdark_test.csv')
parser.add_argument("--root_dir",type=str,default='../CNN_synth_testset/seeingdark/')
parser.add_argument("--batch_size",type=int,default=1)

args = parser.parse_args()

model = Det_RN50()
checkpoint_dir = args.checkpoint
model.load_weights(checkpoint_dir)

optimizer = tf.optimizers.Adam(lr = 1e-4)
loss=tf.keras.losses.BinaryCrossentropy(from_logits = True)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

img_idx = pd.read_csv(args.img_index)

Test_gen = TestDataGenerator(file_index = img_idx, root_dir = args.root_dir, batch_size = args.batch_size)

print('\n\nModel Loaded:{}'.format(checkpoint_dir))
print('\nTesting on:{}\n'.format(args.root_dir))
loss,acc = model.evaluate(Test_gen)