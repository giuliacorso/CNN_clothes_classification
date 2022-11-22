import numpy as np
import torch
import time

from matplotlib import pyplot as plt

from cloth_dataset import ClothDataset
from torch.utils.data import DataLoader
from model import ClothModel
from PIL import Image

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torchvision.transforms as transforms
import cv2
from PIL import ImageOps


def create_confusion_matrix(dataloader_test):
	model = ClothModel()
	model.load_state_dict(torch.load(r"C:\Users\Serena\PycharmProjects\clothes_classifier\ClothClassifier.bin"))
	model.eval()

	y_pred = []
	y_true = []

	# iterate over test data
	for inputs, labels in dataloader_test:
		output = model(inputs)  # Feed Network

		output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
		y_pred.extend(output)  # Save Prediction

		labels = labels.data.cpu().numpy()
		y_true.extend(labels)  # Save Truth

	# constant for classes
	classes = ('dresses', 'upper body', 'lower body')

	# Build confusion matrix
	cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
	df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=0), index=[i for i in classes], columns=[i for i in classes])
	plt.figure(figsize=(12, 7))
	sn.heatmap(df_cm, annot=True)
	plt.savefig(osp.join(args.result_dir, 'confusion_matrix.jpg'))
	plt.imshow(df_cm), plt.show()


def test_real_image(args):
	classes = ['dresses', 'upper body', 'lower body']
	model = ClothModel()
	model.load_state_dict(torch.load(args.checkpoint))
	model.eval()

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	real_image = Image.open(args.real_image).convert('RGB')
	real_image = ImageOps.exif_transpose(real_image)  # toglie la rotazione
	real_image = real_image.resize((192, 256))
	plt.imshow(real_image), plt.show()
	real_image = transform(real_image)
	real_image = torch.unsqueeze(real_image, 0)

	output = model(real_image)
	print(output)
	print("Label predetta: ", classes[output.argmax()])
