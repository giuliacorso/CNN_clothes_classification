from cloth_mask import create_cloth_mask
import os
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
from argument_parser import get_conf


def main_preprocessing(args):
	category = ['dresses', 'upper_body', 'lower_body']
	dataroot = args.dataroot

	for c in category:
		source = osp.join(dataroot, c)
		dest = osp.join(dataroot, c)

		for image in os.listdir(osp.join(source, 'images')):
			create_cloth_mask(image, source, dest)


if __name__ == '__main__':
	args = get_conf()
	main_preprocessing(args)
