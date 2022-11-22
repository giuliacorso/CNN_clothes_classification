from data_augmentation import augment_image
import os
import os.path as osp
from argument_parser import get_conf


if __name__ == '__main__':
	args = get_conf()
	dataroot = args.dataroot

	category = ['dresses', 'upper_body', 'lower_body']

	for c in category:
		d = osp.join(dataroot, c)
		filename = osp.join(d, 'train.txt')
		with open(filename, 'r') as f:
			lines = f.readlines()

		for line in lines:
			im_name = line.strip()
			augment_image(args, d, im_name, 'train')

		filename = osp.join(d, 'test.txt')
		with open(filename, 'r') as f:
			lines = f.readlines()

		for line in lines:
			im_name = line.strip()
			augment_image(args, d, im_name, 'test')
