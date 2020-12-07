from trainer import train

#specify checkpoint path and start epoch. Seed for debugging.
checkpt_path = 'train_model/debug-cp-3.ckpt'
start = 3

train(data_dir = '../Copy_of_progan_train/train/', train_split = 0.1, val_split = 0.0001, small_sample = True, checkpoint = checkpt_path, 
	start_epoch = start, batch_size = 256, epoch = 30, seed = 8)