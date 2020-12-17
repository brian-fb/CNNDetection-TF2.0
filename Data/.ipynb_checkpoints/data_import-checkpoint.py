import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from Data.DataPipe import DataGenerator

# Use small_sample to choose if we want to split the whole data set into two exclusive training/validation set.
def image_generator(train_dir = '../e4040-proj-data/', val_dir = '../progan_val/',train_index='image_names.csv',val_index='validation.csv', batch_size = 64):

	df = pd.read_csv(train_index)
	val_df = pd.read_csv(val_index)
	#train_df, val_df = train_test_split(df, test_size=split)
	train_df = df.sample(frac=1).reset_index(drop = True)
	val_df = val_df.sample(frac=1).reset_index(drop = True)
	print(val_df)
    
	training_gen = DataGenerator(file_index = train_df, root_dir = train_dir, batch_size = batch_size)
	val_gen = DataGenerator(file_index = val_df, root_dir = val_dir, batch_size = batch_size)
    


	return training_gen, val_gen