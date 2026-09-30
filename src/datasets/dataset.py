from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# (channels, image_size)
DATASET_INFO = {
    "cifar10": (3, 32),
    "mnist": (1, 32),
    "fashion_mnist": (1, 32),
}


def get_dataloader(name="cifar10", batch_size=128, train=True, root="data", num_workers=2):
    
    name = name.lower()
    if name not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {name}")

    channels, size = DATASET_INFO[name]

    tfms = [transforms.Resize(size)]
    if train and name == "cifar10":
        tfms.append(transforms.RandomHorizontalFlip())
    tfms += [
        transforms.ToTensor(),                       # [0, 1]
        transforms.Normalize([0.5] * channels, [0.5] * channels),  # [-1, 1]
    ]
    transform = transforms.Compose(tfms)

    if name == "cifar10":
        ds = datasets.CIFAR10(root, train=train, download=True, transform=transform)
    elif name == "mnist":
        ds = datasets.MNIST(root, train=train, download=True, transform=transform)
    else:
        ds = datasets.FashionMNIST(root, train=train, download=True, transform=transform)

    return DataLoader(ds, batch_size=batch_size, shuffle=train,num_workers=num_workers, drop_last=train, pin_memory=True)