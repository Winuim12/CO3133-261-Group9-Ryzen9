def find_errors(labels, predictions):
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have the same length.")

    errors = []

    for index, (true_label, predicted_label) in enumerate(
        zip(labels, predictions)
    ):
        if true_label != predicted_label:
            errors.append({
                "index": index,
                "true_label": true_label,
                "predicted_label": predicted_label,
            })

    return errors