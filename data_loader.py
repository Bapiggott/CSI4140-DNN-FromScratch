import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=128, augment=False):
    """
    Returns train and test DataLoaders for the CIFAR-10 dataset.
    If augment=True, apply basic data augmentations.
    """
    # Define transformations
    transform_list = []
    if augment:
        transform_list += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomVerticalFlip(),

            # Add more if desired
        ]
    transform_list += [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
git 
    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, augment=False)
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)  # Should be [128, 3, 32, 32], [128]