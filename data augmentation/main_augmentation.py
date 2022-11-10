from data_augmentation import augment_image
import os
import os.path as osp


if __name__ == '__main__':
	dataroot = r"C:\Users\Serena\PycharmProjects\clothes_classifier\classifier_dataset"

	category = ['dresses', 'upper_body', 'lower_body']

	for c in category:
		print(c)
		print("Train")
		d = osp.join(dataroot, c)
		filename = osp.join(d, 'train_pairs.txt')
		with open(filename, 'r') as f:
			lines = f.readlines()

		for line in lines:
			_, im_name = line.strip().split()
			augment_image(d, im_name, 'train')

		print("Test")
		filename = osp.join(d, 'test_pairs.txt')
		with open(filename, 'r') as f:
			lines = f.readlines()

		for line in lines:
			_, im_name = line.strip().split()
			augment_image(d, im_name, 'test')
