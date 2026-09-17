#test_dataset.py
'''
Run: python -m pytest tests/test_dataset.py
'''

from src.data.dataset import get_fashion_mnist_datasets

def test_fashion_mnist_datasets():
    train_dataset, val_dataset, test_dataset = get_fashion_mnist_datasets(
        "data/raw/fashion-mnist"
    )

    assert len(train_dataset) == 60000
    assert len(val_dataset) == 60000
    assert len(test_dataset) == 10000