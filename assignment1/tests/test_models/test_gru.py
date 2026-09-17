"""
Unit tests cho RecurrentClassifier (LSTM/GRU) trong rnn.py.
 
Chạy với:
    python -m pytest tests/test_models/test_gru.py -v
"""

import pytest
import torch
 
from src.models.rnn import RecurrentClassifier
 
 
BATCH_SIZE = 8
IMG_SIZE = 28
NUM_CLASSES = 10
 
 
@pytest.fixture
def sample_batch():
    """Batch ảnh giả lập giống Fashion-MNIST: (B, 1, H, W)."""
    return torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
 
 
@pytest.mark.parametrize("mode", ["row", "col", "patch"])
@pytest.mark.parametrize("cell_type", ["lstm", "gru"])
def test_output_shape(sample_batch, mode, cell_type):
    """Output logits phải có shape (B, num_classes)."""
    model = RecurrentClassifier(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        cell_type=cell_type,
        mode=mode,
        patch_size=4,
        hidden_size=32,
    )
    out = model(sample_batch)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"[{cell_type}-{mode}] sai shape output: {out.shape}"
    )
 
 
@pytest.mark.parametrize("mode", ["row", "col", "patch"])
@pytest.mark.parametrize("cell_type", ["lstm", "gru"])
def test_output_no_nan(sample_batch, mode, cell_type):
    """Output không được chứa NaN."""
    model = RecurrentClassifier(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        cell_type=cell_type,
        mode=mode,
        patch_size=4,
        hidden_size=32,
    )
    out = model(sample_batch)
    assert not torch.isnan(out).any(), f"[{cell_type}-{mode}] có NaN trong output"
 
 
@pytest.mark.parametrize(
    "mode,expected_seq_len,expected_input_size",
    [
        ("row", 28, 28),
        ("col", 28, 28),
        ("patch", 49, 16),  # (28/4)^2 = 49 patches, patch_size^2 = 16
    ],
)
def test_sequence_dimensions(mode, expected_seq_len, expected_input_size):
    """seq_len và input_size phải đúng theo từng mode."""
    model = RecurrentClassifier(
        img_size=IMG_SIZE, mode=mode, patch_size=4, hidden_size=16,
    )
    assert model.seq_len == expected_seq_len
    assert model.input_size == expected_input_size
 
 
def test_bidirectional_output_shape(sample_batch):
    """Model bidirectional vẫn phải cho ra đúng shape output."""
    model = RecurrentClassifier(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        cell_type="gru",
        mode="row",
        hidden_size=16,
        bidirectional=True,
    )
    out = model(sample_batch)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)
 
 
def test_multi_layer_with_dropout(sample_batch):
    """Model nhiều layer + dropout vẫn hoạt động và không có NaN."""
    model = RecurrentClassifier(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        cell_type="lstm",
        mode="patch",
        patch_size=7,
        hidden_size=16,
        num_layers=2,
        dropout=0.2,
    )
    out = model(sample_batch)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)
    assert not torch.isnan(out).any()
 
 
def test_invalid_cell_type_raises():
    """cell_type không hợp lệ phải raise ValueError."""
    with pytest.raises(ValueError):
        RecurrentClassifier(cell_type="rnn_vanilla")
 
 
def test_invalid_mode_raises():
    """mode không hợp lệ phải raise ValueError."""
    with pytest.raises(ValueError):
        RecurrentClassifier(mode="diagonal")
 
 
def test_patch_size_not_divisible_raises():
    """img_size không chia hết cho patch_size (ở mode='patch') phải raise lỗi."""
    with pytest.raises(ValueError):
        RecurrentClassifier(img_size=28, mode="patch", patch_size=5)
 