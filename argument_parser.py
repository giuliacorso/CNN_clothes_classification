import argparse  # API for parsing elements from command line


def get_conf(train=True):
	# create the object that holds all the information necessary to parse the command line into Python data types
	parser = argparse.ArgumentParser()
	# Fill the ArgumentParser with information about program arguments
	parser.add_argument("--result_dir", type=str, default="")
	parser.add_argument("--extract_metrics", action='store_true', help='test and extract metrics')
	parser.add_argument("--testing", action='store_true', help='test')
	parser.add_argument("--save_images", action='store_true', help='test')
	parser.add_argument("--exp_name_force", type=str, default="")
	parser.add_argument("--opt_name", type=str, default="")

	parser.add_argument("--name", type=str, default="TOMO")
	parser.add_argument("--stage", type=str, default="UNet")
	parser.add_argument("--ext_folder", type=str, default="test")
	parser.add_argument('-b', '--batch_size', type=int, default=8)
	parser.add_argument('-bs', '--batch_size_cloth', type=int, default=8)
	parser.add_argument('-j', '--workers', type=int, default=15)
	parser.add_argument('--load', action='store_true', help='load_checkpoint')

	parser.add_argument('--exp_name_extract', type=str, default="exp")
	parser.add_argument('--test_images', type=str)
	parser.add_argument('--add_title', type=str, default="")

	parser.add_argument("--dataroot", type=str, default="/work/CucchiaraYOOX2019/students/DressCode")
	parser.add_argument("--phase", default="train")
	parser.add_argument("--category", default='all', type=str)
	parser.add_argument("--height", type=int, default=256)
	parser.add_argument("--width", type=int, default=192)
	parser.add_argument("--radius", type=int, default=5)
	parser.add_argument("--grid_size", type=int, default=5)
	parser.add_argument('--tensorboard_dir', type=str, default='/work/cvcs_2022_group11/tensorboard_dir',
						help='save tensorboard infos')
	parser.add_argument('--checkpoint_dir', type=str, default='/work/cvcs_2022_group11/checkpoint_dir',
						help='save checkpoint infos')
	parser.add_argument("--epoches", type=int, default=150)
	parser.add_argument("--step", type=int, default=100000)
	parser.add_argument("--display_count", type=int, default=1000)
	parser.add_argument('--model_checkpt', type=str, default='/home/davide/Documents/checkpoints/TOMO/tps', help='model_checkpt')

	if train:
		parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs to train')
		parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
		parser.add_argument('--beta2', type=float, default=0.99, help='momentum term of adam')
		parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
		parser.add_argument('--beta1d', type=float, default=0.5, help='momentum term of adam')
		parser.add_argument('--beta2d', type=float, default=0.999, help='momentum term of adam')

	# inspect the command line, convert each argument to the appropriate type and then invoke the appropriate action
	args = parser.parse_args()
	print(args)
	return args



