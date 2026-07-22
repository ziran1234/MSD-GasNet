from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = Path(r"D:\E-nose_DataSet\Gas_Recognition_DataSet\gas_response_data2")
SUPPORTED_GASES = {
    "acetone",
    "benzaldehyde",
    "butyl acetate",
    "dimethylbenzene",
    "methanol"
}
CV2_COLORMAP = 6


def build_output_image(normalized_matrix: np.ndarray, img_size: int) -> np.ndarray:
    """Apply the same color map and resizing strategy used by the existing converter."""

    final_img = cv2.applyColorMap(normalized_matrix, CV2_COLORMAP)
    final_img = cv2.resize(final_img, (img_size, img_size))
    return final_img


def normalize_sequence(data: np.ndarray) -> np.ndarray:
    """Min-max normalize one response segment."""

    data = data.astype(np.float32, copy=True)
    min_value = np.min(data)
    max_value = np.max(data)
    if max_value == min_value:
        return np.zeros_like(data, dtype=np.float32)
    return (data - min_value) / (max_value - min_value)


def create_gasf(data: np.ndarray) -> np.ndarray:
    """Build a GASF matrix from one one-dimensional response segment."""

    normalized_data = normalize_sequence(data)
    cosine_term = normalized_data[:, None] * normalized_data[None, :]
    sine_term = np.sqrt(1 - normalized_data[:, None] ** 2) * np.sqrt(1 - normalized_data[None, :] ** 2)
    return cosine_term - sine_term


def normalize_gasf_matrix(gasf_matrix: np.ndarray) -> np.ndarray:
    """Scale a GASF matrix into uint8 for image saving."""

    min_value = np.min(gasf_matrix)
    max_value = np.max(gasf_matrix)
    if max_value == min_value:
        return np.zeros_like(gasf_matrix, dtype=np.uint8)
    normalized_matrix = (gasf_matrix - min_value) / (max_value - min_value) * 255
    return normalized_matrix.astype(np.uint8)


def _resolve_gas_dir(base_dir: Path, gas_name: str) -> Path:
    if gas_name not in SUPPORTED_GASES:
        raise ValueError(
            "Unsupported gas_name: {}. Expected one of {}.".format(
                gas_name,
                sorted(SUPPORTED_GASES),
            )
        )

    gas_dir = base_dir / gas_name
    if not gas_dir.exists():
        raise FileNotFoundError("Gas directory does not exist: {}".format(gas_dir))

    raw_data_dir = gas_dir / "raw_data"
    if not raw_data_dir.exists():
        raise FileNotFoundError("raw_data directory does not exist: {}".format(raw_data_dir))

    return gas_dir


def _build_segment_ranges(data_length: int, start_index: int, step_index: Sequence[int]) -> List[Tuple[int, int]]:
    if start_index < 0:
        raise ValueError("start_index must be greater than or equal to 0.")
    if start_index >= data_length:
        raise ValueError("start_index must be smaller than the length of the selected data.")
    if not step_index:
        raise ValueError("step_index cannot be empty.")

    segment_ranges = []
    current_start = start_index
    for step in step_index:
        if step <= 0:
            raise ValueError("Each value in step_index must be greater than 0.")
        current_end = current_start + int(step)
        if current_end > data_length:
            break
        segment_ranges.append((current_start, current_end))
        current_start = current_end

    if not segment_ranges:
        raise ValueError("No valid segments could be generated with the provided start_index and step_index.")

    return segment_ranges


def _read_selected_column(csv_path: Path, skip_rows: int, column_index: int) -> np.ndarray:
    data_frame = pd.read_csv(csv_path, header=None, skiprows=skip_rows)
    if data_frame.empty:
        raise ValueError("No data found after applying skip_rows to: {}".format(csv_path))

    column_count = data_frame.shape[1]
    if column_index < 0 or column_index >= column_count:
        raise IndexError(
            "column_index {} is out of range for {}. The file has {} columns.".format(
                column_index,
                csv_path,
                column_count,
            )
        )

    return data_frame.iloc[:, column_index].to_numpy(dtype=np.float32)


def _save_gasf_images_from_data(
    selected_data: np.ndarray,
    output_dir: Path,
    start_index: int,
    step_index: Sequence[int],
    img_size: int,
    start_number: int,
) -> Tuple[List[str], int]:
    segment_ranges = _build_segment_ranges(
        data_length=len(selected_data),
        start_index=start_index,
        step_index=step_index,
    )

    saved_paths = []
    current_number = start_number
    for segment_start, segment_end in segment_ranges:
        gasf_matrix = create_gasf(selected_data[segment_start:segment_end])
        normalized_matrix = normalize_gasf_matrix(gasf_matrix)
        final_img = build_output_image(normalized_matrix, img_size=img_size)

        image_path = output_dir / "{}.png".format(current_number)
        cv2.imwrite(str(image_path), final_img)
        saved_paths.append(str(image_path))
        current_number += 1

    return saved_paths, current_number


def convert_single_csv_to_gasf_images(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    skip_rows: int = 0,
    column_index: int = 0,
    start_index: int = 0,
    step_index: Sequence[int] = (200,),
    img_size: int = 120,
    start_number: int = 0,
) -> List[str]:
    """Convert one CSV file into one or more GASF images based on cumulative segment lengths."""

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_data = _read_selected_column(
        csv_path=csv_path,
        skip_rows=skip_rows,
        column_index=column_index,
    )

    saved_paths, _ = _save_gasf_images_from_data(
        selected_data=selected_data,
        output_dir=output_dir,
        start_index=start_index,
        step_index=step_index,
        img_size=img_size,
        start_number=start_number,
    )

    return saved_paths


def convert_gas_csvs_to_gasf(
    gas_name: str,
    skip_rows: int = 0,
    column_index: int = 0,
    start_index: int = 0,
    step_index: Sequence[int] = (200,),
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    img_size: int = 120,
) -> List[str]:
    """Convert all CSV files under both train_set and test_set into GASF images."""

    base_dir = Path(base_dir)
    gas_dir = _resolve_gas_dir(base_dir=base_dir, gas_name=gas_name)
    raw_data_dir = gas_dir / "raw_data"

    saved_paths = []
    split_to_output = {
        "train_set": gas_dir / "train_set_img",
        "test_set": gas_dir / "test_set_img",
    }

    for split_name, output_dir in split_to_output.items():
        csv_dir = raw_data_dir / split_name
        if not csv_dir.exists():
            raise FileNotFoundError("CSV directory does not exist: {}".format(csv_dir))

        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        current_number = 0
        for csv_file in csv_files:
            selected_data = _read_selected_column(
                csv_path=csv_file,
                skip_rows=skip_rows,
                column_index=column_index,
            )
            current_paths, current_number = _save_gasf_images_from_data(
                selected_data=selected_data,
                output_dir=output_dir,
                start_index=start_index,
                step_index=step_index,
                img_size=img_size,
                start_number=current_number,
            )
            saved_paths.extend(current_paths)

    if not saved_paths:
        raise FileNotFoundError("No CSV files were found under train_set or test_set for gas {}.".format(gas_name))

    return saved_paths


if __name__ == "__main__":
    convert_gas_csvs_to_gasf(
        gas_name="methanol",
        skip_rows=4,
        column_index=9,
        start_index=550,
        step_index=[1200,1200,1150,1150,1150,1250,1150,1150,1150,1150],
        img_size=120,
    )
