import torch


class InfiniteDataLoader:
    """
    A simple infinite dataloader replacement.
    This avoids potential hanging issues with the original custom infinite sampler.
    """

    def __init__(self, dataset, weights, batch_size, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = 0

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights,
                replacement=True,
                num_samples=batch_size
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False
        )
        self.iterator = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class FastDataLoader:
    """
    Standard finite dataloader for evaluation.
    """

    def __init__(self, dataset, batch_size, num_workers):
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
