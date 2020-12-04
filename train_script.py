from trainer import train

#specify checkpoint path and start epoch. Seed for debugging.
checkpt_path = None
start = 0

train(data_dir = '../e4040-proj-data', train_split = 0.999, val_split = 0.0001, small_sample = True, checkpoint = checkpt_path, 
	start_epoch = start, batch_size = 256, epoch = 30, seed = 8)