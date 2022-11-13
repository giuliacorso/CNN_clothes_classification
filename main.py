import torch
import time

from argument_parser import get_conf
from cloth_dataset import ClothDataset
from torch.utils.data import DataLoader
from model import ClothModel

#prova

def train(args):

    epochs = 10
    batch_size = 8
    lr = 0.001

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset e Dataloader
    dataset_train = ClothDataset(args, phase='train')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_test = ClothDataset(args, phase='test')
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ClothModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_steps = len(dataloader_train.dataset) // batch_size

    h = {
        "train_loss": [],
        "train_accuracy": [],
    }

    print("[INFO] training the network...")

    start_time = time.time()

    for e in range(epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        for (images, labels) in dataloader_train:
            #(images, labels) = (images.to(device), labels.to(device))
            labels = labels.reshape(8)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_correct += (output.argmax(1) == labels).type(torch.float).sum().item()

        avg_train_loss = total_train_loss / train_steps
        train_correct = train_correct / len(dataloader_train.dataset)

        #h["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        h["train_loss"].append(avg_train_loss)
        h["train_accuracy"].append(train_correct)

        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avg_train_loss, train_correct))

        #print("[INFO] testing the network...")

        test_correct = 0
        with torch.no_grad():
            for (images, labels) in dataloader_test:
                labels = labels.reshape(8)
                output = model(images)
                test_correct += (output.argmax(1) == labels).type(torch.float).sum().item()

        test_correct = test_correct / len(dataloader_test.dataset)
        print("Test accuracy: {:.4f}".format(test_correct))

    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        end_time - start_time))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_conf()

    print("Images width: ", args.width, "\nImages height: ", args.height)

    train(args)



