from grabcut import create_cloth_mask
import os
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt


def main_preprocessing(source, dest):
	category = ['dresses', 'upper_body', 'lower_body']

	for c in category:
		source_dir = osp.join(source, c)
		dest_dir = osp.join(dest, c)

		for image in os.listdir(osp.join(source_dir, 'images')):
			print(image)
			if image[7] == '1':  # cloth image -> creo cloth mask
				create_cloth_mask(image, source_dir, dest_dir)


if __name__ == '__main__':
	source_d = "/work/CucchiaraYOOX2019/students/DressCode"
	dest_d = "/work/cvcs_2022_group11/masks"

	main_preprocessing(source_d, dest_d)
