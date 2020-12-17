from trainer import train
import argparse
#specify checkpoint path and start epoch. Seed for debugging.
checkpt_path = None

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir",type=str,default='../Copy_of_progan_train/train/')
parser.add_argument("--val_dir",type=str,default='../progan_val/')
parser.add_argument("--small_sample",type=bool,default=False)
parser.add_argument("--checkpoint",type=str,default=None)
parser.add_argument("--start_epoch",type=int,default=0)
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--epoch",type=int,default=30)
parser.add_argument("--train_index", type=str, default=)

args = parser.parse_args()

train(train_dir = args.train_dir, val_dir = args.val_dir, small_sample = args.small_sample, checkpoint = args.checkpoint,
	start_epoch = args.start_epoch, batch_size = args.batch_size, epoch = args.epoch, seed = 8)