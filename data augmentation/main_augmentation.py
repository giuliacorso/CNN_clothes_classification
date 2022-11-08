from data_augmentation import augment_image
import os
import os.path as osp


if __name__ == '__main__':
	dataroot = r"C:\Users\Serena\PycharmProjects\clothes_classifier\classifier_dataset"

	category = ['dresses', 'upper_body', 'lower_body']

	augment_image(osp.join(dataroot, 'dresses'), '053700_1.jpg')

	for c in category:
		d = osp.join(dataroot, c)

		for image in os.listdir(osp.join(d, 'images')):
			print(image)
			if image[7] == '1':
				augment_image(d, image)
