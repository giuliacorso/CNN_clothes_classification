import numpy as np
import torch
import time

from matplotlib import pyplot as plt

from argument_parser import get_conf
from cloth_dataset import ClothDataset
from torch.utils.data import DataLoader
from model import ClothModel


def eval_acc(model, dataloader):
	total = len(dataloader.dataset)
	correct = 0

	with torch.no_grad():
		for (images, labels) in dataloader:
			labels = labels.reshape(8)
			output = model(images)
			correct += (output.argmax(1) == labels).type(torch.float).sum().item()

	return correct / total


def train_function(args):
	epochs = args.epochs
	batch_size = args.batch_size
	lr = 0.001

	# Dataset e Dataloader
	dataset_train = ClothDataset(args, phase='train')
	dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

	dataset_test = ClothDataset(args, phase='test')
	dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

	model = ClothModel()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.CrossEntropyLoss()
	train_steps = len(dataloader_train.dataset) // batch_size

	print("[INFO] training the network...")

	accuracy_train = []
	accuracy_test = []
	start_time = time.time()

	for e in range(epochs):
		model.train()
		total_train_loss = 0
		for j, (images, labels) in enumerate(dataloader_train):
			# (images, labels) = (images.to(device), labels.to(device))
			labels = labels.reshape(batch_size)
			optimizer.zero_grad()
			output = model(images)
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()
			total_train_loss += loss.item()

			if j % 500 == 0:
				print("Epoch ", e + 1, ", step ",  j * 8)
				train_acc = eval_acc(model, dataloader_train)
				test_acc = eval_acc(model, dataloader_test)
				accuracy_test.append(test_acc)
				accuracy_train.append(train_acc)
				print("Accuracy Train: ", train_acc)
				print("Accuracy Test:  ", test_acc)

		avg_train_loss = total_train_loss / train_steps
		train_acc = eval_acc(model, dataloader_train)
		accuracy_train.append(train_acc)

		print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
		print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avg_train_loss, train_acc))

		# print("[INFO] testing the network...")
		test_acc = eval_acc(model, dataloader_test)
		accuracy_test.append(test_acc)
		print("Test accuracy: {:.4f}".format(test_acc))

	end_time = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		end_time - start_time))
	torch.save(model.state_dict(), args.chackpoint)

	plt.plot(np.linspace(1, epochs, epochs*4), accuracy_train, label='train accuracy')
	plt.plot(np.linspace(1, epochs, epochs*4), accuracy_test, label='test accuracy')
	plt.title('training and testing accuracy')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend()
	plt.savefig(osp.join(args.result_dir, 'accuracy_graphic.jpg'))
	plt.show()
