#factory.py

from src.models.cnn import CNNClassifier
from src.models.linear import LinearClassifier
from src.models.mlp import MLPClassifier
from src.models.rnn import RecurrentClassifier
from src.models.transformer import VisionTransformerClassifier


MODEL_REGISTRY = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "cnn": CNNClassifier,
    "rnn": RecurrentClassifier,
    "transformer": VisionTransformerClassifier,
}

def create_model(model_name, model_parameters=None):
    normalized_name = model_name.lower()

    if normalized_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model name: {model_name}")

    model_class = MODEL_REGISTRY[normalized_name]
    parameters = model_parameters or {}

    return model_class(**parameters)