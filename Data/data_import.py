import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from Data.DataPipe import DataGenerator

# Use small_sample to choose if we want to split the whole data set into two exclusive training/validation set.
def image_generator(train_split = 0.9, val_split = 0.01, split = 0.1, data_dir = '../e4040-proj-data/', small_sample = True, seed = None):

	df = pd.read_csv('image_names.csv')
	train_df, val_df = train_test_split(df, test_size=split)
	train_df = train_df.reset_index(drop = True)
	val_df = val_df.reset_index(drop = True)
    
    
	training_gen = DataGenerator(file_index = df, root_dir = data_dir)
	val_gen = DataGenerator(file_index = df, root_dir = data_dir)
    

    
# 	if small_sample:
# 		val_gen = val_gen.flow_from_dataframe(df, data_dir, 
# 	                                                x_col = 'file', y_col = 'label',
# 	                                                subset = 'validation' , shuffle = True, 
# 	                                        class_mode = 'binary', batch_size = 256, seed = seed)

# 	else:
# 		val_gen = img_gen.flow_from_dataframe(df, data_dir, 
# 	                                                x_col = 'file', y_col = 'label',
# 	                                                subset = 'validation' , shuffle = True, 
# 	                                        class_mode = 'binary', batch_size = 256, seed= seed)

	return training_gen, val_gen