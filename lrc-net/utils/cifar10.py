import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


def load_cifar10_dataset(batch_size, thresholds):
    # 1
    trans = [transforms.ToTensor()]
    if thresholds == 3:
        trans.append(transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0)))
    elif thresholds == 31:
        trans.append(transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0)))
    train_transform = transforms.Compose(trans)
    test_transform = transforms.Compose(trans)

    # 2
    # train_trans = [
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ]
    # test_trans = [transforms.ToTensor()]
    # if thresholds == 3:
    #     train_trans.append(transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0)))
    #     test_trans.append(transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0)))
    # elif thresholds == 31:
    #     train_trans.append(transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0)))
    #     test_trans.append(transforms.Lambda(lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0)))
   
    # train_transform = transforms.Compose(train_trans)
    # test_transform = transforms.Compose(test_trans)
    
    train_set = CIFAR10('./data-cifar10', train=True, download=True, transform=train_transform)
    test_set = CIFAR10('./data-cifar10', train=False, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                              drop_last=True)
    input_dim_dataset = 3 * 32 * 32 * thresholds
    num_classes = 10
    return train_loader, test_loader, input_dim_dataset, num_classes

