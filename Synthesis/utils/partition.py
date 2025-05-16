import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def save_partition(partitions, base_dir, prefix):
    os.makedirs(base_dir, exist_ok=True)
    for user_idx, subset in enumerate(partitions):
        user_dir = os.path.join(base_dir, f"{prefix}{user_idx}")
        os.makedirs(user_dir, exist_ok=True)
        class_dirs = [os.path.join(user_dir, str(i)) for i in range(10)]
        for class_dir in class_dirs:
            os.makedirs(class_dir, exist_ok=True)

        loader = DataLoader(subset, batch_size=1, shuffle=False)
        class_counts = [0] * 10

        for idx, (img, label) in enumerate(loader):
            label = label.item()
            save_path = os.path.join(user_dir, str(label), f"{class_counts[label]:05d}.png")
            save_image(img, save_path)
            class_counts[label] += 1

# Partition functions
def partition_pe(dataset):
    num_partitions = 10
    size = len(dataset) // num_partitions
    return [torch.utils.data.Subset(dataset, range(i*size, (i+1)*size)) for i in range(num_partitions)]

def partition_privimage(dataset):
    num_partitions = 5
    size = len(dataset) // num_partitions
    return [torch.utils.data.Subset(dataset, range(i*size, (i+1)*size)) for i in range(num_partitions)]

# Main
if __name__ == "__main__":
    import torch

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    pe_partitions = partition_pe(cifar10_train)
    privimage_partitions = partition_privimage(cifar10_train)

    save_partition(pe_partitions, base_dir="pe_pridata", prefix="usr")
    save_partition(privimage_partitions, base_dir="privimage_pridata", prefix="usr")

    # print("Partitioned data saved successfully.")
