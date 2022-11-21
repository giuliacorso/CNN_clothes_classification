import torch
from torch import nn
import torch.nn.functional as F


class ClothModel(nn.Module):
	def __init__(self):
		super(ClothModel, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 4 * 3, 100)
		self.fc2 = nn.Linear(100, 3)

	def forward(self, input):
		output = F.relu(self.bn1(self.conv1(input)))
		output = F.relu(self.bn2(self.conv2(output)))
		output = self.pool1(output)
		output = F.relu(self.bn3(self.conv3(output)))
		output = F.relu(self.bn4(self.conv4(output)))
		output = self.pool2(output)
		output = output.view(-1, 64 * 4 * 3)
		output = F.relu(self.fc1(output))
		output = self.fc2(output)

		return output


