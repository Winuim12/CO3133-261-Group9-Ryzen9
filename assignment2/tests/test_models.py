import torch
import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN

def test_cnn_classifier_output_shape():
    model = BaselineCNN()
    images = torch.randn(8, 3, 224, 224)

    logits = model(images)

    assert logits.shape == (8, 30)

def test_cnn_classifier_backward_pass():
    model = BaselineCNN()
    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 30, (8,))

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert loss.ndim >= 0
    assert model.classifier[1].weight.grad is not None
