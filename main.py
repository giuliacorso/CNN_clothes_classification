import numpy as np
import torch
import time

from matplotlib import pyplot as plt

from argument_parser import get_conf
from cloth_dataset import ClothDataset
from torch.utils.data import DataLoader
from model import ClothModel

from train import train_function, eval_acc


def train(args):

    epochs = 10
    batch_size = 8
    lr = 0.001

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset e Dataloader
    dataset_train = ClothDataset(phase='train')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_test = ClothDataset(phase='test')
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ClothModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_steps = len(dataloader_train.dataset) // batch_size

    print("[INFO] training the network...")

    start_time = time.time()

    accuracy_train = []
    accuracy_test = []

    for e in range(epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        for j, (images, labels) in enumerate(dataloader_train):
            #(images, labels) = (images.to(device), labels.to(device))
            labels = labels.reshape(batch_size)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_correct += (output.argmax(1) == labels).type(torch.float).sum().item()

            if (j+1) % 500 == 0:
                print("J = ", j+1, "Accuracy attuale = ", train_correct/(j*8))

        avg_train_loss = total_train_loss / train_steps
        train_correct = train_correct / len(dataloader_train.dataset)
        accuracy_train.append(train_correct)

        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avg_train_loss, train_correct))

        #print("[INFO] testing the network...")

        test_correct = 0
        with torch.no_grad():
            for (images, labels) in dataloader_test:
                labels = labels.reshape(batch_size)
                output = model(images)
                test_correct += (output.argmax(1) == labels).type(torch.float).sum().item()

        test_correct = test_correct / len(dataloader_test.dataset)
        accuracy_test.append(test_correct)
        print("Test accuracy: {:.4f}".format(test_correct))

    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        end_time - start_time))
    torch.save(model.state_dict(), r"C:\Users\Serena\PycharmProjects\clothes_classifier\ClothClassifier.bin")
    print(accuracy_train, accuracy_test)

    plt.plot(np.linspace(1, epochs, epochs), accuracy_train, label='train accuracy')
    plt.plot(np.linspace(1, epochs, epochs), accuracy_test, label='test accuracy')
    plt.title('training and testing accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_conf()

    print("Images width: ", 192, "\nImages height: ", 256)

    #train(args)
    train_function(args)



