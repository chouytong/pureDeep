import numpy as np

from src.metrics import classification_metrics


def test_classification_metrics_known_values() -> None:
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        num_classes=2,
    )
    assert metrics["accuracy"] == 0.75
    assert np.isclose(metrics["macro_recall"], 0.75)
    assert np.isclose(metrics["per_class_precision"][0], 1.0)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
