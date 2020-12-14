from trainer import train

#specify checkpoint path and start epoch. Seed for debugging.
checkpt_path = None
start = 0

train(train_dir = '../Copy_of_progan_train/train/', val_dir = '../progan_val/',  train_split = 0.1, val_split = 0.0001, small_sample = False, checkpoint = checkpt_path, 
	start_epoch = start, batch_size = 64, epoch = 30, seed = 8)