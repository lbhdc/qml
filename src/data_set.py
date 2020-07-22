import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist(*, batch_size=1, data_dir="/", n_samples=100, shuffle=True, split=0.7):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = f"{this_dir}/..{data_dir}"

    Xy = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Only select two classes to make this a binary problem
    mask = np.append(
        np.where(Xy.targets == 0)[0][:n_samples],
        np.where(Xy.targets == 1)[0][:n_samples],
    )

    Xy.data = Xy.data[mask]
    Xy.targets = Xy.targets[mask]

    return DataLoader(Xy, batch_size=batch_size, shuffle=shuffle)