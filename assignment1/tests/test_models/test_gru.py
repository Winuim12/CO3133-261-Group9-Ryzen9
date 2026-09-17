import torch
import torch.nn as nn

from src.models.rnn import RecurrentClassifier

def test_rnn_classifier_output_shape():
    model = RecurrentClassifier()
    images = torch.randn(8, 1, 28, 28)

    logits = model(images)

    assert logits.shape == (8, 10)

def test_rnn_classifier_backward_pass():
    model = RecurrentClassifier()
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert loss.ndim >= 0