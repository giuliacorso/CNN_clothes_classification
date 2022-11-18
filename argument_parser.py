import argparse  # API for parsing elements from command line


def get_conf(train=True):
	# create the object that holds all the information necessary to parse the command line into Python data types
	parser = argparse.ArgumentParser()
	# Fill the ArgumentParser with information about program arguments
	parser.add_argument("--dataroot", type=str, default="", help = 'dataroot of dataset')
	parser.add_argument("--phase", type=str, default="train", help = 'train or test')
	parser.add_argument("--backgrounds", type=str, default="", help='dataroot of backgrounds')
	parser.add_argument("--confusion_matrix", action='store_true', help='calculate confusion matrix')
	parser.add_argument('--model_checkpt', type=str, default='', help='dataroot for model state dict')
	parser.add_argument('--test_real_image', type=str, default='', help='path of real image')

	parser.add_argument("--result_dir", type=str, default="")
	parser.add_argument("--extract_metrics", action='store_true', help='test and extract metrics')
	parser.add_argument("--save_images", action='store_true', help='test')

	parser.add_argument('-b', '--batch_size', type=int, default=8)

	parser.add_argument("--category", default='all', type=str)
	parser.add_argument("--height", type=int, default=256)
	parser.add_argument("--width", type=int, default=192)
	parser.add_argument("--epoches", type=int, default=10)

	# inspect the command line, convert each argument to the appropriate type and then invoke the appropriate action
	args = parser.parse_args()
	print(args)
	return args



