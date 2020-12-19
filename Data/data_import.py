import tensorflow as tf
import pandas as pd
from Data.DataPipe import DataGenerator

# Use small_sample to choose if we want to split the whole data set into two exclusive training/validation set.
def image_generator(train_dir = '../e4040-proj-data/', val_dir = '../progan_val/',train_index='image_names.csv',val_index='validation.csv', batch_size = 64, blur_prob=0, jpeg_prob=0):

	df = pd.read_csv(train_index)

	#train_df, val_df = train_test_split(df, test_size=split)
	train_df = df.sample(frac=1).reset_index(drop = True)   # Firstly shuffle the whole data index to ensure unbiased training
	
	if val_dir == 'None' or val_index == 'None':
		val_gen = None

	else:

		val_df = pd.read_csv(val_index)
		val_df = val_df.sample(frac=1).reset_index(drop = True)
		val_gen = DataGenerator(file_index = val_df, root_dir = val_dir, batch_size = batch_size, 
                            blur_prob=0, jpeg_prob=0)
    
	training_gen = DataGenerator(file_index = train_df, root_dir = train_dir, batch_size = batch_size, 
                                 blur_prob=blur_prob, jpeg_prob=jpeg_prob)
	

	return training_gen, val_gen