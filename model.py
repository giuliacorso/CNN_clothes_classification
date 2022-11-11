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

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=4, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(12)
		self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=4, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(12)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(24)
		self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=4, stride=2, padding=1)
		self.bn4 = nn.BatchNorm2d(24)
		self.fc1 = nn.Linear(24 * 8 * 6, 100)
		self.fc2 = nn.Linear(100, 3)

	def forward(self, input):

		output = F.relu(self.bn1(self.conv1(input)))
		output = F.relu(self.bn2(self.conv2(output)))
		output = self.pool(output)
		output = F.relu(self.bn3(self.conv3(output)))
		output = F.relu(self.bn4(self.conv4(output)))
		output = output.view(-1, 24 * 8 * 6)
		#print(output.shape)
		output = F.relu(self.fc1(output))
		#print(output.shape)
		output = self.fc2(output)
		#print(output.shape)

		return output


