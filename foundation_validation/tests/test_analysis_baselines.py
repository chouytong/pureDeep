from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import torch

from src.analysis.common import build_demographic_preprocessor
from src.analysis.features import (
    ACTIVITIES,
    extract_subject_features,
    feature_schema,
    minirocket_subject_tensor,
    signal_feature_vector,
)
from src.analysis.simple_cnn import SimpleSubjectCNN


def test_feature_extraction_is_deterministic_finite_and_label_independent(tmp_path):
    root = tmp_path / "processed"
    rng = np.random.default_rng(42)
    for activity in ACTIVITIES:
        for wrist in ("left", "right"):
            path = root / "signals" / activity / "001" / f"{wrist}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, rng.normal(size=(6, 32)).astype(np.float32), allow_pickle=False)
    first = extract_subject_features(root, "001")
    second = extract_subject_features(root, "001")
    schema = feature_schema()
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert len(first) == len(schema["columns"])
    assert schema["label_dependency"] is False
    assert "label" not in inspect.signature(extract_subject_features).parameters


def test_signal_feature_schema_is_fixed_and_finite():
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000))
    first = signal_feature_vector(signal)
    second = signal_feature_vector(signal.copy())
    assert first == second
    assert len(first) == 28
    assert np.isfinite(first).all()


def test_demographic_preprocessing_fits_train_only():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, np.nan], "height": [100.0, 110.0, 120.0], "weight": [50.0, 60.0, 70.0], "gender": ["f", "m", None], "handedness": ["r", "r", "l"]}
    )
    validation = pd.DataFrame(
        {"age": [1000.0], "height": [999.0], "weight": [999.0], "gender": ["unseen"], "handedness": ["unseen"]}
    )
    preprocessor = build_demographic_preprocessor(["age", "height", "weight"], ["gender", "handedness"])
    transformed_train = preprocessor.fit_transform(train)
    transformed_validation = preprocessor.transform(validation)
    numeric = preprocessor.named_transformers_["numeric"]
    assert numeric.named_steps["imputer"].statistics_[0] == 15.0
    assert numeric.named_steps["scaler"].mean_[0] == 15.0
    assert transformed_validation.shape[1] == transformed_train.shape[1]
    assert np.isfinite(transformed_validation).all()


def test_minirocket_representation_keeps_subject_as_sample(tmp_path):
    root = tmp_path / "processed"
    subjects = ["001", "002"]
    for subject_index, subject in enumerate(subjects):
        for activity_index, activity in enumerate(ACTIVITIES):
            for wrist_index, wrist in enumerate(("left", "right")):
                path = root / "signals" / activity / subject / f"{wrist}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                values = np.full((6, 16 + activity_index), subject_index + wrist_index + activity_index, dtype=np.float32)
                np.save(path, values, allow_pickle=False)
    first = minirocket_subject_tensor(root, subjects, target_length=12)
    second = minirocket_subject_tensor(root, subjects, target_length=12)
    assert first.shape == (2, 11 * 2 * 6, 12)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_simple_cnn_forward_backward_subject_output():
    model = SimpleSubjectCNN(dropout=0.2)
    x = torch.randn(2, 11, 2, 6, 64)
    lengths = torch.full((2, 11), 64, dtype=torch.long)
    mask = torch.ones(2, 11, dtype=torch.bool)
    logits = model(x, lengths, mask)
    assert logits.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
