"""
Unit tests cho VisionTransformerClassifier (và PatchEmbedding) trong transformer.py.

Chạy với:
    python -m pytest tests/test_models/test_transformer.py -v
"""

import pytest
import torch
 
from src.models.transformer import PatchEmbedding, VisionTransformerClassifier
 
 
BATCH_SIZE = 8
IMG_SIZE = 28
NUM_CLASSES = 10
 
 
@pytest.fixture
def sample_batch():
    """Batch ảnh giả lập giống Fashion-MNIST: (B, 1, H, W)."""
    return torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
 
 
@pytest.mark.parametrize("patch_size", [4, 7])
def test_output_shape(sample_batch, patch_size):
    """Output logits phải có shape (B, num_classes)."""
    model = VisionTransformerClassifier(
        img_size=IMG_SIZE,
        in_channels=1,
        num_classes=NUM_CLASSES,
        patch_size=patch_size,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.1,
    )
    out = model(sample_batch)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"[patch_size={patch_size}] sai shape output: {out.shape}"
    )
 
 
@pytest.mark.parametrize("patch_size", [4, 7])
def test_output_no_nan(sample_batch, patch_size):
    """Output không được chứa NaN."""
    model = VisionTransformerClassifier(
        img_size=IMG_SIZE,
        patch_size=patch_size,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
    )
    out = model(sample_batch)
    assert not torch.isnan(out).any(), f"[patch_size={patch_size}] có NaN trong output"
 
 
@pytest.mark.parametrize(
    "patch_size,expected_num_patches",
    [
        (4, 49),   # (28/4)^2
        (7, 16),   # (28/7)^2
        (14, 4),   # (28/14)^2
    ],
)
def test_patch_embedding_num_patches(patch_size, expected_num_patches):
    """PatchEmbedding phải tính đúng số lượng patch."""
    patch_embed = PatchEmbedding(
        img_size=IMG_SIZE, patch_size=patch_size, in_channels=1, embed_dim=16,
    )
    assert patch_embed.num_patches == expected_num_patches
 
 
def test_patch_embedding_output_shape(sample_batch):
    """PatchEmbedding phải trả về (B, num_patches, embed_dim)."""
    embed_dim = 16
    patch_embed = PatchEmbedding(
        img_size=IMG_SIZE, patch_size=4, in_channels=1, embed_dim=embed_dim,
    )
    out = patch_embed(sample_batch)
    assert out.shape == (BATCH_SIZE, 49, embed_dim)
 
 
def test_cls_token_and_pos_embed_shapes():
    """cls_token và pos_embed phải có shape phù hợp với num_patches + 1."""
    embed_dim = 32
    patch_size = 4
    model = VisionTransformerClassifier(
        img_size=IMG_SIZE,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
    )
    num_patches = (IMG_SIZE // patch_size) ** 2
    assert model.cls_token.shape == (1, 1, embed_dim)
    assert model.pos_embed.shape == (1, num_patches + 1, embed_dim)
 
 
def test_img_size_not_divisible_raises():
    """img_size không chia hết cho patch_size phải raise ValueError."""
    with pytest.raises(ValueError):
        VisionTransformerClassifier(img_size=28, patch_size=5)
 
 
def test_embed_dim_not_divisible_by_heads_raises():
    """embed_dim không chia hết cho num_heads phải raise ValueError."""
    with pytest.raises(ValueError):
        VisionTransformerClassifier(embed_dim=30, num_heads=4)
 
 
def test_deeper_model_output_shape(sample_batch):
    """Model với nhiều layer/head hơn vẫn phải cho ra đúng shape và không NaN."""
    model = VisionTransformerClassifier(
        img_size=IMG_SIZE,
        patch_size=4,
        embed_dim=64,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4.0,
        dropout=0.1,
    )
    out = model(sample_batch)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)
    assert not torch.isnan(out).any()
 