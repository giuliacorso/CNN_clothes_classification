import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image


def augment_image(args, dataroot, im_name, phase):
	# apro l'immagine
	image = cv2.imread(osp.join(dataroot, 'images', im_name))
	image = cv2.resize(image, (192, 256))
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	# apro la maschera
	mask = cv2.imread(osp.join(dataroot, 'cloth_masks', im_name.replace('.jpg', '.png')))
	mask = cv2.resize(mask, (192, 256))

	# estraggo i nomi di background
	bg_path = args.backgrounds
	backgrounds = [bg for bg in os.listdir(bg_path)]

	for i in range(4):
		# apro un background randomico
		index = random.randint(0, len(backgrounds) - 1)
		background = cv2.imread(osp.join(bg_path, backgrounds.pop(index)))
		background = cv2.resize(background, (192, 256))
		background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

		# applico una prospettiva a caso
		input_pts = np.float32([[0, 0], [191, 0], [0, 255], [191, 255]])
		output_pts = np.float32([[random.randint(0, 40), random.randint(0, 60)],
								[random.randint(151, 191), random.randint(0, 60)],
								[random.randint(0, 40), random.randint(195, 255)],
								[random.randint(151, 191), random.randint(195, 255)]])
		#print("\nOutput points of perspective\n", output_pts)
		matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
		mod_image = cv2.warpPerspective(image, matrix, (192, 256))
		mod_mask = cv2.warpPerspective(mask, matrix, (192, 256))
		#plt.imshow(mod_image), plt.show()

		# flippo l'immagine
		flip = random.randint(0, 1)
		if flip:
			mod_image = cv2.flip(mod_image, 1)
			mod_mask = cv2.flip(mod_mask, 1)
			#print("\nFlip applied")
			#plt.imshow(mod_image), plt.show()

		# traslazione
		translation_matrix = np.array([
			[1, 0, random.randint(-45, 45)],
			[0, 1, random.randint(-60, 60)]
		], dtype=np.float32)
		# print("\nTranslation with matrix\n", translation_matrix)
		mod_image = cv2.warpAffine(src=mod_image, M=translation_matrix, dsize=(192, 256))
		mod_mask = cv2.warpAffine(src=mod_mask, M=translation_matrix, dsize=(192, 256))
		# plt.imshow(mod_image), plt.show()

		# rotazione e scala
		center = (96, 128)
		angle = random.uniform(-45., 45.)
		scale = random.uniform(1.00, 1.40)
		#print("\nRotation with angle ", angle, " and scale ", scale)
		rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
		mod_image = cv2.warpAffine(src=mod_image, M=rotate_matrix, dsize=(192, 256))
		mod_mask = cv2.warpAffine(src=mod_mask, M=rotate_matrix, dsize=(192, 256))
		#plt.imshow(mod_image), plt.show()

		# applico il background all'immagine trasformata
		output = np.where(mod_mask == 0, background, mod_image)
		#plt.imshow(output), plt.show()

		# salvo l'immagine e la mask
		output_name = im_name.split('.')[0] + "_" + str(i) + ".jpg"
		output = Image.fromarray(output)
		output.save(osp.join(dataroot, 'images', output_name))
		out_mask = Image.fromarray(mod_mask)
		out_mask.save(osp.join(dataroot, 'cloth_masks', output_name.replace('.jpg', '.png')))

		# aggiorno i file txt
		if phase == 'train':
			filename = osp.join(dataroot, 'train.txt')
		else:
			filename = osp.join(dataroot, 'test.txt')

		with open(filename, 'a') as file_object:
			file_object.write(output_name + "\n")
