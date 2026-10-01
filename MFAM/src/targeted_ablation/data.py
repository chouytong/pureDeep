from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.analysis.common import canonical_sha256


@dataclass(frozen=True)
class HandcraftedCache:
    matrix: np.ndarray
    subject_ids: tuple[str, ...]
    labels: np.ndarray
    subject_to_index: Mapping[str, int]


def load_handcrafted_cache(path: str | Path) -> HandcraftedCache:
    payload = np.load(Path(path), allow_pickle=False)
    required = {"X", "subject_ids", "labels"}
    if set(payload.files) != required:
        raise ValueError(f"Unexpected handcrafted cache arrays: {payload.files}")
    matrix = np.asarray(payload["X"], dtype=np.float64)
    subject_ids = tuple(str(value) for value in payload["subject_ids"].tolist())
    labels = np.asarray(payload["labels"], dtype=np.int64)
    if matrix.shape != (len(subject_ids), 4928):
        raise ValueError(f"Expected handcrafted matrix [subjects,4928], got {matrix.shape}")
    if labels.shape != (len(subject_ids),):
        raise ValueError("Handcrafted labels shape mismatch")
    if len(subject_ids) != len(set(subject_ids)) or not np.isfinite(matrix).all():
        raise ValueError("Handcrafted cache IDs or values are invalid")
    return HandcraftedCache(
        matrix=matrix,
        subject_ids=subject_ids,
        labels=labels,
        subject_to_index={subject: index for index, subject in enumerate(subject_ids)},
    )


@dataclass
class FoldStatisticalTransform:
    scaler: StandardScaler
    pca: PCA
    transformed: dict[str, np.ndarray]
    metadata: dict[str, Any]


def fit_fold_statistical_transform(
    cache: HandcraftedCache,
    train_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    *,
    components: int = 32,
    random_state: int = 42,
) -> FoldStatisticalTransform:
    train_ids = tuple(str(value) for value in train_subjects)
    validation_ids = tuple(str(value) for value in validation_subjects)
    if set(train_ids) & set(validation_ids):
        raise ValueError("R2 train and validation subjects overlap")
    unknown = (set(train_ids) | set(validation_ids)) - set(cache.subject_ids)
    if unknown:
        raise ValueError(f"Subjects missing from handcrafted cache: {sorted(unknown)}")
    train_indices = [cache.subject_to_index[value] for value in train_ids]
    selected_ids = train_ids + validation_ids
    selected_indices = [cache.subject_to_index[value] for value in selected_ids]
    train_matrix = cache.matrix[train_indices]
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled_train = scaler.fit_transform(train_matrix)
    pca = PCA(
        n_components=int(components),
        whiten=False,
        svd_solver="full",
        random_state=int(random_state),
    )
    pca.fit(scaled_train)
    transformed_matrix = pca.transform(scaler.transform(cache.matrix[selected_indices]))
    if transformed_matrix.shape != (len(selected_ids), int(components)):
        raise AssertionError("R2 transformed matrix shape mismatch")
    if not np.isfinite(transformed_matrix).all():
        raise ValueError("R2 transformed features contain non-finite values")
    transformed = {
        subject: transformed_matrix[index].astype(np.float32, copy=True)
        for index, subject in enumerate(selected_ids)
    }
    metadata = {
        "method": "inner_train StandardScaler then inner_train PCA",
        "feature_count_input": int(cache.matrix.shape[1]),
        "feature_count_output": int(components),
        "scaler_fit_subject_count": len(train_ids),
        "pca_fit_subject_count": len(train_ids),
        "fit_subject_ids_sha256": canonical_sha256(list(train_ids)),
        "validation_subject_ids_sha256": canonical_sha256(list(validation_ids)),
        "validation_labels_used_for_fit": False,
        "outer_test_features_transformed": False,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "pca_whiten": False,
        "pca_svd_solver": "full",
        "random_state": int(random_state),
    }
    return FoldStatisticalTransform(
        scaler=scaler,
        pca=pca,
        transformed=transformed,
        metadata=metadata,
    )


class StatisticalFeatureDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        base: Dataset[dict[str, object]],
        transformed: Mapping[str, np.ndarray],
    ) -> None:
        self.base = base
        self.transformed = {
            str(subject): np.asarray(value, dtype=np.float32)
            for subject, value in transformed.items()
        }
        self.subject_ids = tuple(str(value) for value in getattr(base, "subject_ids"))
        missing = set(self.subject_ids) - set(self.transformed)
        if missing:
            raise ValueError(f"Missing transformed R2 features: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = dict(self.base[index])
        subject = str(item["subject_id"])
        item["statistical_features"] = torch.from_numpy(
            self.transformed[subject].copy()
        )
        return item
