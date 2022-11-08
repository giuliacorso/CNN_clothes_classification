import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image


def augment_image(dataroot, im_name):
	# apro l'immagine
	image = cv2.imread(osp.join(dataroot, 'images', im_name))
	image = cv2.resize(image, (192, 256))

	# apro la maschera
	mask = cv2.imread(osp.join(dataroot, 'cloth_masks', im_name.replace('.jpg', '.png')))
	mask = cv2.resize(mask, (192, 256))

	# estraggo i nomi di background
	bg_path = r"C:\Users\Serena\PycharmProjects\clothes_classifier\classifier_dataset\backgrounds"
	backgrounds = [bg for bg in os.listdir(bg_path)]

	for i in range(3):
		# apro un background a caso
		index = random.randint(0, len(backgrounds) - 1)
		background = cv2.imread(osp.join(bg_path, backgrounds[index]))
		background = cv2.resize(background, (192, 256))

		# applico una prospettiva a caso
		input_pts = np.float32([[0, 0], [191, 0], [0, 255], [191, 255]])
		output_pts = np.float32([[random.randint(0, 50), random.randint(0, 70)],
								[random.randint(141, 191), random.randint(0, 70)],
								[random.randint(0, 50), random.randint(185, 255)],
								[random.randint(141, 191), random.randint(185, 255)]])

		matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
		mod_image = cv2.warpPerspective(image, matrix, (192, 256))
		mod_mask = cv2.warpPerspective(mask, matrix, (192, 256))

		# applico il background all'immagine trasformata
		output = np.where(mod_mask == 0, background, mod_image)

		# salvo
		output = Image.fromarray(output)
		output.save(osp.join(dataroot, 'augmented_images', im_name.replace('.jpg', '.png')))











