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

	def forward(self):
		pass


