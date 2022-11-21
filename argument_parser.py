import argparse  # API for parsing elements from command line


def get_conf():
	# create the object that holds all the information necessary to parse the command line into Python data types
	parser = argparse.ArgumentParser()
	# Fill the ArgumentParser with information about program arguments
	parser.add_argument("--dataroot", type=str, default="", help = 'dataroot of dataset')
	parser.add_argument('--checkpoint', type=str, default='', help='path of load state dict file')
	parser.add_argument('--real_image', type=str, default='', help='path of real image')
	parser.add_argument('--backgrounds', type=str, default='', help='path of backgrounds')
	parser.add_argument("--phase", type=str, default="train", help='train or test')
	parser.add_argument("--confusion_matrix", action='store_true', help='calculate confusion matrix')
	#parser.add_argument("--acc_freq", type=int, default=0, help="how many times we calculate accuracy inside an epoch")
	parser.add_argument("--result_dir", type=str, default='', help='directory for saving results')

	parser.add_argument("--category", default='all', type=str)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument('-b', '--batch_size', type=int, default=8)

	# inspect the command line, convert each argument to the appropriate type and then invoke the appropriate action
	args = parser.parse_args()
	print(args)
	return args



