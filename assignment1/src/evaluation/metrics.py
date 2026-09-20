# metrics.py

from sklearn.metrics import f1_score

def calculate_macro_f1(predictions, targets):
    predictions_array = predictions.detach().cpu().numpy()
    targets_array = targets.detach().cpu().numpy()

    return float(
        f1_score(
            y_true=targets_array,
            y_pred=predictions_array,
            average="macro",
            zero_division=0,
        )
    )

def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )