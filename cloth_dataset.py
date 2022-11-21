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
		self.dataroot = args.dataroot
		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		size = 0
		category = ['dresses', 'upper_body', 'lower_body']
		dataroot_names = []
		cloth_names = []

		for c in category:
			dataroot = osp.join(self.dataroot, c)
			if phase == 'train':
				filename = osp.join(dataroot, 'train.txt')
			else:
				filename = osp.join(dataroot, 'test.txt')

			with open(filename, 'r') as f:
				lines = f.readlines()
				size += len(lines)
				for line in lines:
					c_name = line.strip()
					cloth_names.append(c_name)
					dataroot_names.append(dataroot)

		self.dataroot_names = dataroot_names
		self.cloth_names = cloth_names
		self.size = size

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		c_name = self.cloth_names[index]
		dataroot = self.dataroot_names[index]

		# Cloth image
		# apro l'immagine dell'item
		cloth = Image.open(osp.join(dataroot, 'images', c_name))
		cloth = cloth.resize((self.width, self.height))
		cloth = self.transform(cloth)

		label_name = dataroot.split('\\')[-1]
		if label_name == 'dresses':
			label = torch.tensor([0])
		elif label_name == 'upper_body':
			label = torch.tensor([1])
		else:
			label = torch.tensor([2])

		return cloth, label
