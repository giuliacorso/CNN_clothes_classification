from PIL import Image
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F


from matplotlib.path import Path
import torch
import os
from torch import nn
import numpy as np
import cv2


class ClothModel(nn.Module):
	def __init__(self, args):
		super(ClothModel, self).__init__()
		self.args = args

		self.conv1 = nn.Conv2d(3, 3, 5, 2)
		self.conv2 = nn.Conv2d(3, 3, 3, 2)
		self.pool = nn.MaxPool2d(3, 2)
		self.fc = nn.Linear(420, 3)


	def forward(self,x):
		x = self.conv1(x)
		#print(x.shape)
		x = self.pool(x)
		x = self.conv2(x)
		x = self.pool(x)
		print(x.shape)
		x = self.fc(x)
		return x


