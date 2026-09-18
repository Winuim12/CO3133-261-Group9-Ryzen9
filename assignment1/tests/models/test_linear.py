import torch
import torch.nn as nn

from src.models.linear import LinearClassifier

def test_linear_classifier_output_shape():
    model = LinearClassifier()
    images = torch.randn(8, 1, 28, 28)  

    logits = model(images)
    assert logits.shape == (8, 10)

def test_linear_classifier_backward_pass():
    model = LinearClassifier()
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert loss.ndim >= 0
    assert model.linear.weight.grad is not None