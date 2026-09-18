import torch

def create_optimizer(model, optimizer_name, learning_rate, weight_decay):
    normalized_name = optimizer_name.lower()

    if normalized_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer name: {optimizer_name}")