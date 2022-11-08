import torch

from argument_parser import get_conf
from cloth_dataset import ClothDataset
from torch.utils.data import DataLoader
from model import ClothModel


def main_function(args):
    # Dataset e Dataloader
    dataset_train = ClothDataset(args, phase = 'train')
    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=0)

    dataset_test = ClothDataset(args, phase = 'test')
    dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=0)

    # modello
    x = torch.ones((3,256,192))
    model = ClothModel(args)
    model(x)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_conf()

    print("Images width: ", args.width, "\nImages height: ", args.height)

    main_function(args)



