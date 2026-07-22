from pathlib import Path
from typing import List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_BASE_DIR = Path(r"D:\E-nose_DataSet\Gas_Recognition_DataSet\gas_response_data2")
SUPPORTED_GASES = {
    "acetone",
    "benzaldehyde",
    "butyl acetate",
    "dimethylbenzene",
    "methanol"
}
SUPPORTED_SPLITS = {"train_set", "test_set"}
DEFAULT_RESPONSE_COLUMNS = list(range(9, 17))


def _resolve_csv_folder(base_dir: Path, gas_name: str, dataset_split: str) -> Path:
    if gas_name not in SUPPORTED_GASES:
        raise ValueError(
            f"Unsupported gas_name: {gas_name}. "
            f"Expected one of {sorted(SUPPORTED_GASES)}."
        )

    if dataset_split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported dataset_split: {dataset_split}. "
            f"Expected one of {sorted(SUPPORTED_SPLITS)}."
        )

    csv_folder = base_dir / gas_name / "raw_data" / dataset_split
    if not csv_folder.exists():
        raise FileNotFoundError(f"CSV folder does not exist: {csv_folder}")

    return csv_folder


def _normalize_column_indices(column_indices: Union[int, Sequence[int]]) -> List[int]:
    if column_indices == -1:
        return DEFAULT_RESPONSE_COLUMNS.copy()

    if isinstance(column_indices, int):
        return [column_indices]

    normalized_indices = [int(column_index) for column_index in column_indices]
    if not normalized_indices:
        raise ValueError("column_indices cannot be empty.")

    return normalized_indices


def _validate_column_indices(column_count: int, response_columns: Sequence[int], csv_path: Path) -> None:
    invalid_indices = [
        column_index
        for column_index in response_columns
        if column_index < 0 or column_index >= column_count
    ]
    if invalid_indices:
        raise IndexError(
            f"Column indices {invalid_indices} are out of range for {csv_path}. "
            f"The file has {column_count} columns."
        )


def _add_vertical_lines(
    axes: plt.Axes,
    x_values,
    start_index: int,
    step_index: Sequence[int],
) -> None:
    if start_index < 0:
        raise ValueError("start_index must be greater than or equal to 0.")
    if start_index >= len(x_values):
        raise ValueError("start_index must be smaller than the length of x_values.")
    if not step_index:
        raise ValueError("step_index cannot be empty when draw_vertical_lines is True.")

    current_index = start_index
    axes.axvline(x=x_values[current_index], color="gray", linestyle="--", linewidth=1.2, alpha=0.8)

    for step in step_index:
        if step <= 0:
            raise ValueError("Each value in step_index must be greater than 0.")
        current_index += step
        if current_index >= len(x_values):
            break
        axes.axvline(x=x_values[current_index], color="gray", linestyle="--", linewidth=1.2, alpha=0.8)


def plot_single_csv_curve(
    csv_path: Union[str, Path],
    skip_rows: int = 0,
    column_indices: Union[int, Sequence[int]] = -1,
    draw_vertical_lines: bool = False,
    start_index: int = 0,
    step_index: Sequence[int] = (1,),
    figure_size: Tuple[int, int] = (12, 6),
) -> None:
    """Plot the first-column x-axis against one or more selected response columns."""

    csv_path = Path(csv_path)
    response_columns = _normalize_column_indices(column_indices)

    data_frame = pd.read_csv(csv_path, header=None, skiprows=skip_rows)
    if data_frame.empty:
        raise ValueError(f"No data found after applying skip_rows to: {csv_path}")

    column_count = data_frame.shape[1]
    if column_count == 0:
        raise ValueError(f"No columns found in CSV file: {csv_path}")

    _validate_column_indices(column_count, response_columns, csv_path)

    x_values = data_frame.iloc[:, 0].to_numpy()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    for response_column in response_columns:
        y_values = data_frame.iloc[:, response_column].to_numpy()
        axes.plot(x_values, y_values, linewidth=1.5, label=f"Column {response_column}")

    if draw_vertical_lines:
        _add_vertical_lines(
            axes=axes,
            x_values=x_values,
            start_index=start_index,
            step_index=step_index,
        )

    axes.set_title(csv_path.stem)
    axes.set_xlabel("Column 0")
    axes.set_ylabel("Response")
    axes.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    if len(response_columns) > 1:
        axes.legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_gas_response_curves(
    gas_name: str,
    dataset_split: str,
    skip_rows: int = 4,
    column_indices: Union[int, Sequence[int]] = -1,
    draw_vertical_lines: bool = False,
    start_index: int = 0,
    step_index: Sequence[int] = (1,),
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    figure_size: Tuple[int, int] = (12, 6),
) -> List[str]:
    """Traverse the target folder and plot one figure for each CSV file."""

    base_dir = Path(base_dir)
    csv_folder = _resolve_csv_folder(base_dir=base_dir, gas_name=gas_name, dataset_split=dataset_split)
    csv_files = sorted(csv_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in: {csv_folder}")

    plotted_files = []
    for csv_file in csv_files:
        plot_single_csv_curve(
            csv_path=csv_file,
            skip_rows=skip_rows,
            column_indices=column_indices,
            draw_vertical_lines=draw_vertical_lines,
            start_index=start_index,
            step_index=step_index,
            figure_size=figure_size,
        )
        plotted_files.append(str(csv_file))

    return plotted_files


if __name__ == "__main__":
    plot_gas_response_curves(
        gas_name="acetone",
        dataset_split="train_set",
        skip_rows=4,
        column_indices=9,
        draw_vertical_lines=True,
        start_index=550,
        step_index=[1200,1200,1150,1150,1150,1250,1150,1150,1150,1150],
    )
