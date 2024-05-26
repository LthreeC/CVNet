from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data

Mean = [0.76313736, 0.5474047, 0.57174409]
Std = [0.09055175, 0.11924913, 0.13393524]

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