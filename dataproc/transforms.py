from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from dataproc.config import Mean, Std

def data_process(ROOT, ReSize=256, BatchSize=32):
    normalize = transforms.Normalize(mean=Mean, std=Std)

    # trans
    data_transform = transforms.Compose([transforms.Resize((ReSize, ReSize)),
                                          transforms.ToTensor(),
                                          normalize])
    # data_transform = transforms.Compose([
    #     transforms.Resize((ReSize, ReSize)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.ToTensor(),
    #     normalize
    # ])

    # auto load
    data = ImageFolder(ROOT, transform=data_transform)
    num_classes = len(data.classes)  # get classes

    dataloader = Data.DataLoader(dataset=data,
                                       batch_size=BatchSize,
                                       shuffle=True,
                                       num_workers=0)
    return dataloader, num_classes