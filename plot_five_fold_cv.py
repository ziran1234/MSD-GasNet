import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _build_axes(figure_size, xlabel="epoch", ylabel="Loss"):
    """Create axes styled to match ``plot_train_test_figure``."""

    legend_font = {
        "family": "Arial",
        "style": "normal",
        "size": 40,
        "weight": "bold",
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    bold_font = {"fontname": "Arial", "weight": "bold"}
    axes.spines["bottom"].set_linewidth(5)
    axes.spines["left"].set_linewidth(5)
    axes.spines["right"].set_linewidth(5)
    axes.spines["top"].set_linewidth(5)
    axes.set_xlabel(xlabel, labelpad=3, fontsize=48, **bold_font)
    axes.set_ylabel(ylabel, labelpad=5, fontsize=48, **bold_font)
    axes.tick_params(axis="x", labelsize=34, direction="out", width=4, length=10)
    axes.tick_params(axis="y", labelsize=34, direction="out", width=4, length=10)
    x_label = axes.get_xticklabels()
    [x_label_temp.set_fontweight("bold") for x_label_temp in x_label]
    y_label = axes.get_yticklabels()
    [y_label_temp.set_fontweight("bold") for y_label_temp in y_label]
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    return fig, axes, legend_font


def _validate_csv_columns(dataframe):
    """Validate that the five-fold detail CSV contains the required columns."""

    required_columns = {
        "fold",
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "CSV file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def _plot_metric(dataframe, train_column, val_column, ylabel, save_path):
    """Plot five light fold curves and one dark mean curve for train/validation."""

    fig, axes, legend_font = _build_axes((12, 8), xlabel="Epoch", ylabel=ylabel)
    filtered_dataframe = dataframe.copy()
    filtered_dataframe = filtered_dataframe[pd.to_numeric(filtered_dataframe["fold"], errors="coerce").notna()]
    filtered_dataframe["fold"] = filtered_dataframe["fold"].astype(int)

    train_pivot = (
        filtered_dataframe
        .pivot_table(index="epoch", columns="fold", values=train_column, aggfunc="mean")
        .sort_index()
    )
    val_pivot = (
        filtered_dataframe
        .pivot_table(index="epoch", columns="fold", values=val_column, aggfunc="mean")
        .sort_index()
    )
    epoch_values = train_pivot.index.tolist()

    light_red = "#f3a2a2"
    dark_red = "#8b0000"
    light_blue = "#a8c9ff"
    dark_blue = "#003f8c"

    for fold_id in train_pivot.columns:
        axes.plot(
            epoch_values,
            train_pivot[fold_id].to_numpy(),
            color=light_red,
            linewidth=2.0,
            alpha=0.9,
        )
    for fold_id in val_pivot.columns:
        axes.plot(
            epoch_values,
            val_pivot[fold_id].to_numpy(),
            color=light_blue,
            linewidth=2.0,
            alpha=0.9,
        )

    train_mean = train_pivot.mean(axis=1)
    val_mean = val_pivot.mean(axis=1)
    axes.plot(
        epoch_values,
        train_mean.to_numpy(),
        color=dark_red,
        linewidth=4.0,
        label="train mean",
    )
    axes.plot(
        epoch_values,
        val_mean.to_numpy(),
        color=dark_blue,
        linewidth=4.0,
        label="test mean",
    )
    axes.legend(prop=legend_font)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_five_fold_curves(csv_path):
    """Read one five-fold detail CSV and save loss/accuracy figures."""

    csv_path = Path(csv_path).resolve()
    dataframe = pd.read_csv(csv_path)
    _validate_csv_columns(dataframe)

    output_dir = csv_path.parent
    loss_save_path = output_dir / f"{csv_path.stem}_loss.png"
    accuracy_save_path = output_dir / f"{csv_path.stem}_accuracy.png"

    _plot_metric(
        dataframe=dataframe,
        train_column="train_loss",
        val_column="val_loss",
        ylabel="Loss",
        save_path=loss_save_path,
    )
    _plot_metric(
        dataframe=dataframe,
        train_column="train_accuracy",
        val_column="val_accuracy",
        ylabel="Accuracy",
        save_path=accuracy_save_path,
    )

    print(f"[FiveFoldPlot] Saved loss figure to: {loss_save_path}")
    print(f"[FiveFoldPlot] Saved accuracy figure to: {accuracy_save_path}")
    return str(loss_save_path), str(accuracy_save_path)


def main():
    """Parse CLI arguments and plot five-fold curves from one CSV file."""

    parser = argparse.ArgumentParser(description="Plot five-fold cross-validation curves from one CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the five-fold detail CSV file.")
    args = parser.parse_args()
    plot_five_fold_curves(args.csv_path)


if __name__ == "__main__":
    plot_five_fold_curves(r"D:\Work2_Denoise\GasRecognition\five_fold_cv_results\mine_multiscale_only_GASF_img_five_fold_detail_2026_05_06_15_44_26.csv")
