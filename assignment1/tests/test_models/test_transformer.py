import torch
import torch.nn as nn

from src.models.transformer import VisionTransformerClassifier

def test_transformer_classifier_output_shape():
    model = VisionTransformerClassifier()
    images = torch.randn(8, 1, 28, 28)

    logits = model(images)

    assert logits.shape == (8, 10)

def test_transformer_classifier_backward_pass():
    model = VisionTransformerClassifier()
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert loss.ndim >= 0