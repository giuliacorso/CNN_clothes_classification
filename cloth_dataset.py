import torch
import torch.utils.data as data
import os.path as osp

from PIL import Image
import torchvision.transforms as transforms

class ClothDataset(data.Dataset):

	def __init__(self, args, phase='train'):
		super(ClothDataset, self).__init__()
		self.width = 192
		self.height = 256
		#self.dataroot = r"C:\Users\Serena\PycharmProjects\clothes_classifier\classifier_dataset"
		self.dataroot = r"C:\Users\giuli\OneDrive - Unimore\magistrale\II anno\school in ai\progetto\classifier_dataset"

		size = 0
		category = ['dresses', 'upper_body', 'lower_body']
		dataroot_names = []
		cloth_names = []
		for c in category:
			dataroot = osp.join(self.dataroot, c)
			if phase == 'train':
				filename = osp.join(dataroot, 'train_pairs.txt')
				filename_augmented = osp.join(dataroot, 'augmented_train.txt')
			else:
				filename = osp.join(dataroot, 'test_pairs.txt')
				filename_augmented = osp.join(dataroot, 'augmented_test.txt')
			with open(filename, 'r') as f:
				lines = f.readlines()
				size += len(lines)
				for line in lines:
					_, c_name = line.strip().split()
					cloth_names.append(c_name)
					dataroot_names.append(dataroot)
			with open(filename_augmented, 'r') as f:
				lines_augmented = f.readlines()
				size += len(lines_augmented)
				for line in lines_augmented:
					_, c_name = line.strip().split()
					cloth_names.append(c_name)
					dataroot_names.append(dataroot)

		self.dataroot_names = dataroot_names
		print(len(dataroot_names))
		self.cloth_names = cloth_names
		print(len(cloth_names))
		self.size = size

		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		c_name = self.cloth_names[index]
		dataroot = self.dataroot_names[index]

		# Cloth image
		# apro l'immagine dell'item
		check = "_1_"
		if check in c_name:
			cloth = Image.open(osp.join(dataroot, 'augmented_images', c_name))
		else:
			cloth = Image.open(osp.join(dataroot, 'images', c_name))
		cloth = cloth.resize((self.width, self.height))
		cloth = self.transform(cloth)

		label_name = dataroot.split('\\')[-1]
		if label_name == 'dresses':
			label = torch.tensor([0])
		if label_name == 'upper_body':
			label = torch.tensor([1])
		if label_name == 'lower_body':
			label = torch.tensor([2])

		return cloth, label