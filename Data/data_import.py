import tensorflow as tf
import pandas as pd

# Use small_sample to choose if we want to split the whole data set into two exclusive training/validation set.
def image_generator(train_split = 0.9, val_split = 0.01, data_dir = '../e4040-proj-data', small_sample = True):
	img_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split = train_split, rescale = 1./255)
	val_gen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split = val_split, rescale = 1.0/255)

	df = pd.read_csv('image_names.csv')
	df = df.sample(frac=1).reset_index(drop=True)
	df = df.astype({'label': 'str'})

	training_gen = img_gen.flow_from_dataframe(df, data_dir, 
	                                                x_col = 'file', y_col = 'label',
	                                                subset = 'training' , shuffle = True, 
	                                        class_mode = 'binary', batch_size = 256)

	df = pd.read_csv('image_names.csv')
	df = df.sample(frac=1).reset_index(drop=True)

	df = df.astype({'label': 'str'})

	if small_sample:
		val_gen = val_gen.flow_from_dataframe(df, data_dir, 
	                                                x_col = 'file', y_col = 'label',
	                                                subset = 'validation' , shuffle = True, 
	                                        class_mode = 'binary', batch_size = 256)

	else:
		val_gen = img_gen.flow_from_dataframe(df, data_dir, 
	                                                x_col = 'file', y_col = 'label',
	                                                subset = 'validation' , shuffle = True, 
	                                        class_mode = 'binary', batch_size = 256)
	return training_gen, val_gen