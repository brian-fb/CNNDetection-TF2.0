import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,Input,GlobalAveragePooling2D,AveragePooling2D
import datetime
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from Data.data_import import image_generator

def train(data_dir, train_split, val_split, small_sample = False, checkpoint = None, 
		save_path = 'train_model/cp-{epoch:d}.ckpt', batch_size = 256, epoch = 30, start_epoch = 0, seed = None):
	train_gen, val_gen = image_generator(train_split = train_split, val_split = val_split, 
										data_dir = data_dir, small_sample = small_sample, seed = seed)
    
	print(len(train_gen))

	base_model = applications.ResNet50(weights='imagenet', include_top=False)  
	#base_model = applications.xception.Xception(weights='imagenet', include_top=False, input_shape=[resize_H,resize_W,channel]) 

	for layer in base_model.layers[:140]:  # Keep the pretrained params
		   layer.trainable = False
	for layer in base_model.layers[140:]:  # Keep the pretrained params
		   layer.trainable = True
		    
	x = base_model.output  # 
	x = GlobalAveragePooling2D()(x)  
	# x = Dense(1024, activation='relu', name='fc1',kernel_regularizer=keras.regularizers.l2(0.0001))(x)  
	# x = Dropout(0.5)(x)  # Droupout 0.6
	# x = Dense(512, activation='relu', name='fc2',kernel_regularizer=keras.regularizers.l2(0.0001))(x)
	# #x = Dropout(0.5)(x)
	predictions = Dense(1, name='predictions')(x)  
	model = Model(base_model.input,predictions)  
	
	if checkpoint is not None: 
		model.load_weights(checkpoint)
	#Hyperparameters
	optimizer = tf.optimizers.Adam(lr = 1e-4)

	loss=tf.keras.losses.BinaryCrossentropy(from_logits = True)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                 save_weights_only=True,
                                                 verbose=1)

	model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
	steps = len(train_gen)
	model.fit(x = train_gen, validation_data = val_gen, epochs=epoch, shuffle = True, batch_size = batch_size,
	 steps_per_epoch = steps, callbacks = [cp_callback], initial_epoch= start_epoch)

