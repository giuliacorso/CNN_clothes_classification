import torch
import torch.utils.data as data
import os.path as osp


class ClothDataset(data.Dataset):

	def __init__(self, args, phase = 'train'):
		super(ClothDataset, self).__init__()
		self.width = 192
		self.height = 256
		self.dataroot = r"C:\Users\Serena\PycharmProjects\clothes_classifier\classifier_dataset"

		size = 0
		category = ['dresses', 'upper_body', 'lower_body']
		dataroot_names = []
		cloth_names = []
		for c in category:
			dataroot = osp.join(self.dataroot, c)
			if phase == 'train':
				filename = osp.join(dataroot, 'train_pairs.txt')
			else:
				filename = osp.join(dataroot, 'test_pairs_paired.txt')
			with open(filename, 'r') as f:
				size += len(f.readlines())
				for line in f.readlines():
					_, c_name = line.strip().split()
					cloth_names.append(c_name)
					dataroot_names.append(dataroot)

		self.dataroot_names = dataroot_names
		self.cloth_names = cloth_names
		self.size = size

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		c_name = self.c_names[index]
		dataroot = self.dataroot_names[index]

		# Cloth image
		# apro l'immagine dell'item
		cloth = Image.open(osp.join(dataroot, 'images', c_name))
		cloth = cloth.resize((self.width, self.height))

		# Cloth mask
		# apro la cloth mask creata con il grabcut (/work/cvcs_2022_group11/masks/.../cloth_masks/)
		category = osp.split(dataroot)[-1]
		c_mask_path = osp.join(self.dataroot, category, 'cloth_masks', c_name.replace('.jpg', '.png'))
		c_mask = Image.open(c_mask_path)
		c_mask = c_mask.resize((self.width, self.height))
		# applico la cloth_mask al cloth cioè metto lo sfondo bianco (1)
		#cloth[c_mask.repeat(3, 1, 1) == 0.] = 1.

		label = dataroot.split('/')[-1]

		result = {
			"cloth_image" : cloth,
			"cloth_mask" : c_mask,
			"label" : label
		}

		return result
