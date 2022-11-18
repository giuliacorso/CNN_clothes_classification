from cloth_mask import create_cloth_mask
import os
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt


def main_preprocessing(dataroot):
	category = ['dresses', 'upper_body', 'lower_body']

	for c in category:
		source = osp.join(dataroot, c)
		dest = osp.join(dataroot, c)

		for image in os.listdir(osp.join(source, 'images')):
			#print(image)
			create_cloth_mask(image, source, dest)

		print("HO FINITO")


if __name__ == '__main__':
	dataset_dir = r"C:\Users\Serena\PycharmProjects\clothes_classifier\classifier_dataset"
	#dataset_dir = r"C:\Users\giuli\OneDrive - Unimore\magistrale\II anno\school in ai\progetto\classifier_dataset"

	main_preprocessing(dataset_dir)
