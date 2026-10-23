#dataloader.py
'''
Same interface as Assignment 1, plus a seed so the training shuffle order and
the random augmentation in worker processes are reproducible.
'''

import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
