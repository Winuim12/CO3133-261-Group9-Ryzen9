#test_pipeline.py
'''
RUN: python -m pytest tests/test_data/test_pipeline.py
'''
import torch

from src.data.pipeline import create_normalized_datasets

def test_normalized_pipeline_creates_expected_datasets():
    train_subset, val_subset, test_dataset, mean, std = (
        create_normalized_datasets(
            data_dir="data/raw/fashion-mnist",
            validation_size=0.1,
            seed=42,
        )
    )

    assert len(train_subset) == 54_000
    assert len(val_subset) == 6_000
    assert len(test_dataset) == 10_000

    assert train_subset.dataset is not val_subset.dataset

    assert 0.28 < mean < 0.29
    assert 0.34 < std < 0.36

    train_image, train_label = train_subset[0]
    val_image, val_label = val_subset[0]
    test_image, test_label = test_dataset[0]

    assert torch.isfinite(train_image).all()
    assert torch.isfinite(val_image).all()
    assert torch.isfinite(test_image).all()

    assert 0 <= train_label < 10
    assert 0 <= val_label < 10
    assert 0 <= test_label < 10