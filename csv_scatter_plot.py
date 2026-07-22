from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


DISTINCT_COLORS = [
    "#d62728",  # red
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
    "#003f5c",  # dark blue
    "#ffa600",  # amber
]

def _infer_plot_columns(data_frame, label_col_index=None, x_col_index=None, y_col_index=None):
    """Infer the label and coordinate columns from CSV headers, or fall back to explicit indices."""

    if {"true_class", "dim1", "dim2"}.issubset(data_frame.columns):
        return "true_class", "dim1", "dim2"
    if {"true_label", "dim1", "dim2"}.issubset(data_frame.columns):
        return "true_label", "dim1", "dim2"
    if label_col_index is None or x_col_index is None or y_col_index is None:
        raise ValueError(
            "Could not infer plotting columns from CSV headers. Please provide "
            "label_col_index, x_col_index, and y_col_index explicitly."
        )
    column_count = data_frame.shape[1]
    column_indices = [label_col_index, x_col_index, y_col_index]
    for column_index in column_indices:
        if column_index < 0 or column_index >= column_count:
            raise IndexError(
                f"Column index {column_index} is out of range. The CSV only has {column_count} columns."
            )
    return (
        data_frame.columns[label_col_index],
        data_frame.columns[x_col_index],
        data_frame.columns[y_col_index],
    )


def _infer_label_order(data_frame, label_column_name):
    """Return label order that follows true_label when available, otherwise first appearance order."""

    if "true_class" in data_frame.columns and "true_label" in data_frame.columns and label_column_name == "true_class":
        ordered_rows = (
            data_frame[["true_label", "true_class"]]
            .drop_duplicates()
            .sort_values("true_label")
        )
        return ordered_rows["true_class"].tolist()
    if "pred_class" in data_frame.columns and "pred_label" in data_frame.columns and label_column_name == "pred_class":
        ordered_rows = (
            data_frame[["pred_label", "pred_class"]]
            .drop_duplicates()
            .sort_values("pred_label")
        )
        return ordered_rows["pred_class"].tolist()
    return list(pd.unique(data_frame[label_column_name]))


def _build_label_color_map(label_values):
    """Assign one clearly distinct color to each label value."""

    if len(label_values) > len(DISTINCT_COLORS):
        raise ValueError(
            f"The current palette supports up to {len(DISTINCT_COLORS)} classes, got {len(label_values)}."
        )
    return {label_value: DISTINCT_COLORS[index] for index, label_value in enumerate(label_values)}


def generate_horizontal_gas_legend(
    save_path=None,
    csv_path=None,
    figure_size=(18, 1.6),
    fontsize=40,
    markersize=30,
    num_rows=None,
    show=False,
):
    """Generate a standalone horizontal legend figure inferred from one cluster CSV when provided."""

    if csv_path is not None:
        csv_path = Path(csv_path)
        data_frame = pd.read_csv(csv_path)
        label_column_name, _, _ = _infer_plot_columns(data_frame)
        gas_labels = _infer_label_order(data_frame, label_column_name)
    else:
        gas_labels = [
            "acetone",
            "benzaldehyde",
            "butyl acetate",
            "dimethylbenzene",
            "methanol",
        ]

    label_color_map = _build_label_color_map(gas_labels)
    legend_handles = []
    for gas_label in gas_labels:
        color = label_color_map[gas_label]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                label=str(gas_label),
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=markersize,
            )
        )

    if num_rows is None:
        num_rows = 1 if len(gas_labels) <= 6 else 2
    if num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    legend_columns = int((len(gas_labels) + num_rows - 1) // num_rows)
    if num_rows > 1 and figure_size == (18, 1.6):
        figure_size = (18, 1.6 * num_rows)

    fig, axes = plt.subplots(figsize=figure_size)
    fig.patch.set_facecolor("white")
    axes.set_facecolor("white")
    axes.axis("off")
    legend = fig.legend(
        handles=legend_handles,
        loc="center",
        ncol=legend_columns,
        frameon=False,
        fontsize=fontsize,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_fontname("Arial")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(save_path) if save_path is not None else None


def plot_scatter_from_csv(
    csv_path,
    label_col_index=None,
    x_col_index=None,
    y_col_index=None,
    save_path=None,
    figure_size=(10, 8),
    show=False,
):
    """Read one cluster CSV and draw a scatter plot colored by true class content."""

    csv_path = Path(csv_path)
    data_frame = pd.read_csv(csv_path)
    label_column_name, x_column_name, y_column_name = _infer_plot_columns(
        data_frame,
        label_col_index=label_col_index,
        x_col_index=x_col_index,
        y_col_index=y_col_index,
    )

    label_series = data_frame[label_column_name]
    x_series = data_frame[x_column_name]
    y_series = data_frame[y_column_name]
    unique_labels = _infer_label_order(data_frame, label_column_name)
    label_color_map = _build_label_color_map(unique_labels)

    bold_font = {"fontname": "Arial", "weight": "bold"}
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    for label_value in unique_labels:
        class_mask = label_series == label_value
        axes.scatter(
            x_series[class_mask],
            y_series[class_mask],
            c=label_color_map[label_value],
            label=str(label_value),
            s=48,
            alpha=0.9,
        )

    axes.spines["bottom"].set_linewidth(3)
    axes.spines["left"].set_linewidth(3)
    axes.spines["right"].set_linewidth(3)
    axes.spines["top"].set_linewidth(3)

    reduce_method = str(data_frame["reduce_method"].iloc[0]).lower() if "reduce_method" in data_frame.columns else ""
    if reduce_method == "tsne":
        x_label = "t-SNE 1"
        y_label = "t-SNE 2"
    elif reduce_method == "pca":
        x_label = "PCA 1"
        y_label = "PCA 2"
    elif reduce_method == "umap":
        x_label = "UMAP 1"
        y_label = "UMAP 2"
    else:
        x_label = str(x_column_name)
        y_label = str(y_column_name)

    axes.set_xlabel(x_label, labelpad=8, fontsize=40, **bold_font)
    axes.set_ylabel(y_label, labelpad=-16, fontsize=40, **bold_font)
    axes.xaxis.set_major_locator(MultipleLocator(25))
    axes.yaxis.set_major_locator(MultipleLocator(25))
    axes.tick_params(axis="x", labelsize=38, direction="out", width=4)
    axes.tick_params(axis="y", labelsize=38, direction="out", width=4)
    for tick_label in axes.get_xticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_fontname("Arial")
    for tick_label in axes.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_fontname("Arial")

    plt.tight_layout()

    if save_path is None:
        save_path = csv_path.with_suffix(".png")
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(save_path)


def batch_plot_scatter_from_folder(
    folder_path,
    label_col_index=None,
    x_col_index=None,
    y_col_index=None,
    save_dir=None,
    recursive=False,
    show=False,
):
    """Traverse a folder and plot all cluster CSV files, inferring columns from file content when possible."""

    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    csv_pattern = "**/*.csv" if recursive else "*.csv"
    csv_files = sorted(folder_path.glob(csv_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in: {folder_path}")

    output_dir = Path(save_dir) if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for csv_file in csv_files:
        if output_dir is None:
            target_path = csv_file.with_suffix(".png")
        else:
            target_path = output_dir / f"{csv_file.stem}.png"
        saved_path = plot_scatter_from_csv(
            csv_path=csv_file,
            label_col_index=label_col_index,
            x_col_index=x_col_index,
            y_col_index=y_col_index,
            save_path=target_path,
            show=show,
        )
        saved_paths.append(saved_path)

    return saved_paths


if __name__ == "__main__":
    cluster_dir = r"E:\上海师范大学本科生科研工作总结-xyr\cluster_3"
    # batch_plot_scatter_from_folder(
    #     folder_path=cluster_dir,
    #     save_dir=cluster_dir,
    #     recursive=False,
    #     show=False,
    # )
    generate_horizontal_gas_legend(
        csv_path=Path(cluster_dir) / "mine_multiscale_only_GASF_img_tsne_cluster_fold5_2026_07_02_19_46_13.csv",
        save_path=Path(cluster_dir) / "legend.png",
        num_rows= 4
    )
