import tensorflow as tf
from Data.data_import import image_generator
from Network.Det_RN50 import Det_RN50

def train(train_dir, val_dir,train_idx,val_idx small_sample = False,
		checkpoint = None,save_path = 'train_model/model5-cp-{epoch:d}.ckpt',
				batch_size = 64, epoch = 30, start_epoch = 0, seed = None):

	train_gen, val_gen = image_generator(train_dir = train_dir, val_dir = val_dir, train_index=train_idx,
										 val_index=val_idx small_sample = small_sample, seed = seed, batch_size = batch_size)

	model = Det_RN50()
	
	if checkpoint is not None: 
		model.load_weights(checkpoint)

	#Hyperparameters
	optimizer = tf.optimizers.Adam(lr = 1e-4)

	loss=tf.keras.losses.BinaryCrossentropy(from_logits = True)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True, verbose=1)

	model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
	steps = len(train_gen)

	model.fit(x = train_gen, validation_data =val_gen, epochs=epoch, shuffle = True,
	 steps_per_epoch = steps, callbacks = [cp_callback], initial_epoch= start_epoch)

