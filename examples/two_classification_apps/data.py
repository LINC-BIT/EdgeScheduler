import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


def get_caltech256_dataloader(split, batch_size):
    dataset = ImageFolder('/data/zql/datasets/Caltech-256/data/caltech256/256_ObjectCategories/', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    if split == 'train':
        dataset = split_dataset(dataset, int(len(dataset) * 0.8))[0]
    else:
        dataset = split_dataset(dataset, int(len(dataset) * 0.8))[1]
    
    return InfiniteDataLoader(
        dataset,
        weights=None,
        batch_size=batch_size,
        num_workers=4
    )


def get_domainnet_dataloader(split, batch_size):
    dataset = ImageFolder('/data/zql/datasets/domain_net/real', transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    if split == 'train':
        dataset = split_dataset(dataset, int(len(dataset) * 0.8))[0]
    else:
        dataset = split_dataset(dataset, int(len(dataset) * 0.8))[1]
    
    return InfiniteDataLoader(
        dataset,
        weights=None,
        batch_size=batch_size,
        num_workers=4
    )


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, collate_fn=None):
        super().__init__()

        if weights:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=batch_size
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )

        if collate_fn is not None:
            self._infinite_iterator = iter(
                torch.utils.data.DataLoader(
                    dataset,
                    num_workers=num_workers,
                    batch_sampler=_InfiniteSampler(batch_sampler),
                    pin_memory=False,
                    collate_fn=collate_fn
                )
            )
        else:
            self._infinite_iterator = iter(
                torch.utils.data.DataLoader(
                    dataset,
                    num_workers=num_workers,
                    batch_sampler=_InfiniteSampler(batch_sampler),
                    pin_memory=False
                )
            )
        self.dataset = dataset

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError
    

import torch
import os
import numpy as np


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset), f'{n}_{len(dataset)}'

    cache_p = f'{n}_{seed}_{len(dataset)}'
    cache_p = os.path.join(os.path.expanduser(
        '~'), '.domain_benchmark_split_dataset_cache_' + str(cache_p))
    if os.path.exists(cache_p):
        keys_1, keys_2 = torch.load(cache_p)
    else:
        keys = list(range(len(dataset)))
        np.random.RandomState(seed).shuffle(keys)
        keys_1 = keys[:n]
        keys_2 = keys[n:]
        torch.save((keys_1, keys_2), cache_p)
    
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def train_val_split(dataset, split, rate=0.8):
    assert split in ['train', 'val']
    if split == 'train':
        return split_dataset(dataset, int(len(dataset) * rate))[0]
    else:
        return split_dataset(dataset, int(len(dataset) * rate))[1]
