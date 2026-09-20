#error_analysis.py

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

def save_prediction_examples(
    dataset, 
    targets,
    predictions, 
    class_names, 
    mean, 
    std, 
    path, 
    examples_per_group=5, 
    title="Prediction Examples"
):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    targets=targets.detach().cpu()
    predictions=predictions.detach().cpu()

    correct_indices = torch.where(predictions == targets)[0][:examples_per_group]
    incorrect_indices = torch.where(predictions != targets)[0][:examples_per_group]

    figure, axes = plt.subplots(
        2, 
        examples_per_group,
        figsize=(3*examples_per_group, 6),
        squeeze=False,
    )

    groups = [
        ("Correct", correct_indices),
        ("Incorrect", incorrect_indices),
    ]

    for row_index, (group_name, indices) in enumerate(groups):
        for column_index in range(examples_per_group):
            axis = axes[row_index, column_index]
            axis.axis("off")

            if column_index >= len(indices):
                continue

            sample_index = indices[column_index].item()
            image, _ = dataset[sample_index]

            #Reverse normalization for display.
            displayed_image = image * std + mean
            displayed_image = displayed_image.clamp(0, 1)

            target_id = targets[sample_index].item()
            prediction_id = predictions[sample_index].item()

            axis.imshow(displayed_image.squeeze().detach().cpu().numpy(), cmap="gray")

            axis.set_title(
                f"{group_name}\n"
                f"Actual: {class_names[target_id]}\n"
                f"Predicted: {class_names[prediction_id]}"
            )

    figure.suptitle(title)
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)