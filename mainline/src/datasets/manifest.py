from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SENSOR_CHANNELS = ("AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ")
WRIST_ORDER = ("left", "right")
SENSOR_INDICES = {
    "acc": (0, 1, 2),
    "gyro": (3, 4, 5),
    "acc_gyro": (0, 1, 2, 3, 4, 5),
}


@dataclass(frozen=True)
class ManifestRecord:
    """一行代表同一 PADS session 内已经验证过的左右腕记录。"""

    left_path: Path
    right_path: Path
    subject_id: str
    label: int
    activity: str
    pair_id: str
    fold: int | None
    session_index: int | None = None
    record_name: str = ""
    source_condition: str = ""
    left_length: int | None = None
    right_length: int | None = None
    left_sampling_rate: float | None = None
    right_sampling_rate: float | None = None
    max_time_offset_seconds: float | None = None

    @property
    def sample_id(self) -> str:
        return self.pair_id


def _parse_label(raw: str, labels: list[str]) -> int:
    if raw in labels:
        return labels.index(raw)
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"Unknown label {raw!r}; expected one of {labels} or an integer index"
        ) from error
    if not 0 <= value < len(labels):
        raise ValueError(f"Label index {value} is outside [0, {len(labels)})")
    return value


def _optional_int(raw: str | None) -> int | None:
    value = (raw or "").strip()
    return int(value) if value else None


def _optional_float(raw: str | None) -> float | None:
    value = (raw or "").strip()
    return float(value) if value else None


def _resolve_signal_path(raw: str, root: Path, row_number: int, side: str) -> Path:
    value = Path(raw.strip()).expanduser()
    path = (value if value.is_absolute() else root / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest row {row_number} points to missing {side} file: {path}"
        )
    return path


def load_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
    labels: list[str],
    activity: str | None = None,
    fold_column: str = "fold",
) -> list[ManifestRecord]:
    """读取双腕 manifest；拒绝旧的单文件/单腕 manifest。"""
    path = Path(manifest_path).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run scripts/prepare_data.sh first."
        )
    required = {"left_path", "right_path", "subject_id", "label", "activity"}
    records: list[ManifestRecord] = []
    seen_pairs: set[tuple[Path, Path]] = set()
    seen_pair_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Bilateral manifest is missing columns: "
                f"{sorted(missing)}. Regenerate it with prepare_data.sh."
            )
        for row_number, row in enumerate(reader, start=2):
            row_activity = (row.get("activity") or "").strip()
            if activity and row_activity != activity:
                continue
            left_path = _resolve_signal_path(
                row["left_path"], root, row_number, "left"
            )
            right_path = _resolve_signal_path(
                row["right_path"], root, row_number, "right"
            )
            pair_id = (row.get("pair_id") or "").strip()
            if not pair_id:
                raise ValueError(f"Missing pair_id in manifest row {row_number}")
            path_pair = (left_path, right_path)
            if path_pair in seen_pairs:
                raise ValueError(f"Duplicate wrist pair in manifest row {row_number}")
            if pair_id in seen_pair_ids:
                raise ValueError(
                    f"Duplicate pair_id in manifest row {row_number}: {pair_id}"
                )
            seen_pairs.add(path_pair)
            seen_pair_ids.add(pair_id)
            fold_raw = (row.get(fold_column) or "").strip()
            records.append(
                ManifestRecord(
                    left_path=left_path,
                    right_path=right_path,
                    subject_id=row["subject_id"].strip(),
                    label=_parse_label(row["label"].strip(), labels),
                    activity=row_activity,
                    pair_id=pair_id,
                    fold=int(fold_raw) if fold_raw else None,
                    session_index=_optional_int(row.get("session_index")),
                    record_name=(row.get("record_name") or "").strip(),
                    source_condition=(row.get("source_condition") or "").strip(),
                    left_length=_optional_int(row.get("left_length")),
                    right_length=_optional_int(row.get("right_length")),
                    left_sampling_rate=_optional_float(row.get("left_sampling_rate")),
                    right_sampling_rate=_optional_float(row.get("right_sampling_rate")),
                    max_time_offset_seconds=_optional_float(
                        row.get("max_time_offset_seconds")
                    ),
                )
            )
    if not records:
        suffix = f" for activity={activity!r}" if activity else ""
        raise ValueError(f"No bilateral records selected from {path}{suffix}")
    return records


def _load_array(path: Path, delimiter: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        keys = list(archive.keys())
        if len(keys) != 1:
            raise ValueError(
                f"{path} must contain exactly one NPZ array; found keys {keys}"
            )
        array = archive[keys[0]]
    elif suffix in {".csv", ".txt"}:
        array = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported time-series extension: {path.suffix}")
    return np.asarray(array, dtype=np.float32)


def shape_time_series(
    array: np.ndarray,
    input_channels: int,
    path: Path,
    drop_time_column: bool = False,
) -> np.ndarray:
    # PADS TXT 为 [time, Time+6 channels]；统一输出 [channels,time]。
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D array in {path}, got shape {array.shape}")
    if drop_time_column:
        if array.shape[1] == input_channels + 1:
            array = array[:, 1:]
        elif array.shape[0] == input_channels + 1:
            array = array[1:, :]
        else:
            raise ValueError(
                f"Cannot identify a single time column in {path}: {array.shape}"
            )
    if array.shape[0] == input_channels:
        shaped = array
    elif array.shape[1] == input_channels:
        shaped = array.T
    else:
        raise ValueError(
            f"Neither dimension in {path} matches input_channels={input_channels}: "
            f"{array.shape}"
        )
    if not np.isfinite(shaped).all():
        raise ValueError(f"NaN or infinite values found in {path}")
    return np.ascontiguousarray(shaped, dtype=np.float32)


def selected_channel_indices(sensor_mode: str) -> tuple[int, ...]:
    try:
        return SENSOR_INDICES[str(sensor_mode).lower()]
    except KeyError as error:
        raise ValueError(
            f"sensor_mode must be one of {sorted(SENSOR_INDICES)}, got {sensor_mode!r}"
        ) from error


def selected_channel_names(sensor_mode: str) -> list[str]:
    return [SENSOR_CHANNELS[index] for index in selected_channel_indices(sensor_mode)]


def crop_or_pad(
    signal: np.ndarray,
    sequence_length: int,
    crop: str,
    pad_value: float,
) -> np.ndarray:
    """沿最后一个时间维裁剪；支持 [C,T] 和 [2,C,T]。"""
    length = signal.shape[-1]
    if length == sequence_length:
        return signal
    if length > sequence_length:
        if crop == "start":
            start = 0
        elif crop == "center":
            start = (length - sequence_length) // 2
        elif crop == "random":
            start = int(np.random.randint(0, length - sequence_length + 1))
        else:
            raise ValueError(f"Unsupported crop mode: {crop!r}")
        return signal[..., start : start + sequence_length]
    total = sequence_length - length
    left = total // 2 if crop == "center" else 0
    right = total - left
    padding = [(0, 0)] * signal.ndim
    padding[-1] = (left, right)
    return np.pad(signal, padding, constant_values=float(pad_value))


def multi_crop_or_pad(
    signal: np.ndarray,
    sequence_length: int,
    num_crops: int,
    pad_value: float,
) -> np.ndarray:
    """输出 [views,...,time]；成对腕信号共用同一组裁剪起点。"""
    if num_crops < 1:
        raise ValueError("num_crops must be at least 1")
    length = signal.shape[-1]
    if length <= sequence_length:
        padded = crop_or_pad(signal, sequence_length, "center", pad_value)
        return np.repeat(padded[None, ...], num_crops, axis=0)
    starts = np.rint(
        np.linspace(0, length - sequence_length, num=num_crops)
    ).astype(np.int64)
    return np.stack(
        [signal[..., int(start) : int(start) + sequence_length] for start in starts],
        axis=0,
    )


class ManifestTimeSeriesDataset(Dataset[dict[str, object]]):
    """双腕 PADS 数据集，单裁剪输出 [2,C,T]，多裁剪输出 [V,2,C,T]。"""

    def __init__(
        self,
        records: list[ManifestRecord],
        raw_input_channels: int,
        sensor_mode: str,
        wrist_mode: str,
        sequence_length: int | None,
        crop: str,
        pad_value: float,
        delimiter: str,
        drop_time_column: bool,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        num_crops: int = 1,
    ) -> None:
        self.records = records
        self.raw_input_channels = int(raw_input_channels)
        self.sensor_mode = str(sensor_mode).lower()
        self.channel_indices = selected_channel_indices(self.sensor_mode)
        self.channel_names = selected_channel_names(self.sensor_mode)
        self.wrist_mode = str(wrist_mode).lower()
        if self.wrist_mode not in {"left", "right", "bilateral"}:
            raise ValueError("wrist_mode must be left, right, or bilateral")
        self.sequence_length = (
            None if sequence_length is None else int(sequence_length)
        )
        self.crop = crop
        self.pad_value = float(pad_value)
        self.delimiter = delimiter
        self.drop_time_column = bool(drop_time_column)
        self.mean = mean
        self.std = std
        self.num_crops = int(num_crops)
        if self.sequence_length is not None and self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2 when configured")
        if self.crop == "multi" and self.sequence_length is None:
            raise ValueError("multi-crop evaluation requires a fixed sequence_length")
        if self.crop == "multi" and self.num_crops < 1:
            raise ValueError("num_crops must be at least 1 for multi-crop evaluation")

    def __len__(self) -> int:
        return len(self.records)

    def _load_wrist(self, path: Path) -> torch.Tensor:
        shaped = shape_time_series(
            _load_array(path, self.delimiter),
            self.raw_input_channels,
            path,
            drop_time_column=self.drop_time_column,
        )
        return torch.from_numpy(shaped[list(self.channel_indices)])

    def full_signal(self, index: int) -> torch.Tensor:
        record = self.records[index]
        left = self._load_wrist(record.left_path)
        right = self._load_wrist(record.right_path)
        if left.shape != right.shape:
            raise ValueError(
                f"Paired wrist length/channel mismatch for {record.pair_id}: "
                f"left={tuple(left.shape)}, right={tuple(right.shape)}"
            )
        # 固定腕侧顺序为 [left,right]，形状 [2,C,T]。
        return torch.stack((left, right), dim=0)

    def statistics_signal(self, index: int) -> torch.Tensor:
        # 不按 wrist_mode 丢弃腕，保证三种对照可复用同一训练集统计过程。
        return self.full_signal(index)

    def raw_signal(self, index: int) -> torch.Tensor:
        signal = self.full_signal(index).numpy()
        if self.sequence_length is None:
            return torch.from_numpy(np.ascontiguousarray(signal))
        if self.crop == "multi":
            shaped = multi_crop_or_pad(
                signal, self.sequence_length, self.num_crops, self.pad_value
            )
        else:
            shaped = crop_or_pad(
                signal, self.sequence_length, self.crop, self.pad_value
            )
        return torch.from_numpy(np.ascontiguousarray(shaped))

    def _wrist_mask(self) -> torch.Tensor:
        masks = {
            "left": (1.0, 0.0),
            "right": (0.0, 1.0),
            "bilateral": (1.0, 1.0),
        }
        return torch.tensor(masks[self.wrist_mode], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        signal = self.raw_signal(index)
        if self.mean is not None and self.std is not None:
            signal = (signal - self.mean) / self.std
        mask = self._wrist_mask()
        # 占位腕在标准化后清零，且后续不会进入共享编码器。
        mask_shape = (1, 2, 1, 1) if signal.ndim == 4 else (2, 1, 1)
        signal = signal * mask.reshape(mask_shape)
        return {
            "x": signal,
            "wrist_mask": mask,
            "y": torch.tensor(record.label, dtype=torch.long),
            "subject_id": record.subject_id,
            "pair_id": record.pair_id,
            "sample_id": record.pair_id,
            "activity": record.activity,
            "wrist_mode": self.wrist_mode,
            "sensor_mode": self.sensor_mode,
            "left_path": str(record.left_path),
            "right_path": str(record.right_path),
        }
