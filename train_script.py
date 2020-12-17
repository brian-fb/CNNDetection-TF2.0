import tensorflow as tf
from Data.data_import import image_generator
from Network.Det_RN50 import Det_RN50
import argparse
#specify checkpoint path and start epoch. Seed for debugging.
checkpt_path = None

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir",type=str,default='../Copy_of_progan_train/train/')
parser.add_argument("--val_dir",type=str,default='../progan_val/')
parser.add_argument("--blur_prob",type=float,default= 0)
parser.add_argument("--jpeg_prob",type=float,default= 0)
parser.add_argument("--checkpoint",type=str,default=None)
parser.add_argument("--start_epoch",type=int,default=0)
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--epoch",type=int,default=30)
parser.add_argument("--train_index", type=str, default='Img_index/train/progan_train.csv')
parser.add_argument("--val_index", type=str, default='Img_index/val/progan_val.csv')

args = parser.parse_args()

def train(train_dir, val_dir,train_idx,val_idx, small_sample = False,
		checkpoint = None,save_path = 'train_model/model5-cp-{epoch:d}.ckpt',
				batch_size = 64, epoch = 30, start_epoch = 0, seed = None, blur_prob, jpeg_prob):

	train_gen, val_gen = image_generator(train_dir = train_dir, val_dir = val_dir, train_index=train_idx,
										 val_index=val_idx, batch_size = batch_size, blur_prob=blur_prob, jpeg_prob=jpeg_prob)
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

if __name__ == '__main__' :
    train(train_dir = args.train_dir, val_dir = args.val_dir, train_idx = args.train_index, 
          val_idx = args.val_index, checkpoint = args.checkpoint,start_epoch = args.start_epoch, 
          batch_size = args.batch_size, epoch = args.epoch, blur_prob = args.blur_prob, jpeg_prob = args.jpeg_prob, seed = None)