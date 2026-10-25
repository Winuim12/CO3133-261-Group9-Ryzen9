import random

import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_repeats_random_sequences():
    set_seed(42)
    first_python = random.random()
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)

    set_seed(42)

    assert random.random() == first_python
    np.testing.assert_array_equal(np.random.rand(3), first_numpy)
    assert torch.equal(torch.rand(3), first_torch)