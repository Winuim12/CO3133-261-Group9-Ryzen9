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

def top_confused_pairs(confusion_matrix, classes, top=10):
    """Most frequent (true, predicted) mistakes, with the rate within the true class."""
    pairs = []

    for true_index, row in enumerate(confusion_matrix):
        total = sum(row)
        for predicted_index, count in enumerate(row):
            if true_index != predicted_index and count:
                pairs.append({
                    "true": classes[true_index],
                    "predicted": classes[predicted_index],
                    "count": int(count),
                    "rate_of_true_class": count / total,
                })

    pairs.sort(key=lambda pair: (-pair["count"], pair["true"], pair["predicted"]))
    return pairs[:top]


def errors_by_confidence(confidences, correct, bins=(0.5, 0.8)):
    """Count wrong predictions per confidence band, e.g. <0.5, 0.5-0.8, >=0.8."""
    edges = (0.0, *bins, 1.0)
    counts = {}

    for low, high in zip(edges, edges[1:]):
        name = f"<{high}" if low == 0.0 else (f">={low}" if high == 1.0 else f"{low}-{high}")
        counts[name] = sum(
            1 for confidence, is_correct in zip(confidences, correct)
            if not is_correct and low <= confidence and (confidence < high or high == 1.0)
        )

    return counts
