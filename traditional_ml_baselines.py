import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from Dataset import GeneratedGasImageDataset, MyDataset, gas_display_name, gas_label, gas_name_array


DEFAULT_MYDATASET_BASE_DATA_PATH = r"D:\E-nose_DataSet\Gas_Recognition_DataSet\gas_response_data"
DEFAULT_GENERATED_BASE_DATA_PATH = r"D:\E-nose_DataSet\Gas_Recognition_DataSet\gas_response_data2"
RESULTS_DIR_NAME = "traditional_ml_results"
CONFUSION_DIR_NAME = "traditional_ml_confusion_matrix"
CLUSTER_DIR_NAME = "traditional_ml_cluster"
RESULTS_MULTI_DIR_NAME = "traditional_ml_results_2"
CONFUSION_MULTI_DIR_NAME = "traditional_ml_confusion_matrix_2"
CLUSTER_MULTI_DIR_NAME = "traditional_ml_cluster_2"
model_list = [
    "logistic_regression",
    "random_forest",
    "qda",
    "knn"]
SUPPORTED_CLASSIFIERS = {
    "logistic_regression",
    "random_forest",
    "qda",
    "gaussian_nb",
    "knn",
}
SUPPORTED_DATASETS = {
    "mydataset",
    "generated",
    "generatedgasimagedataset",
}
TRADITIONAL_ML_ALGORITHM_CONFIGS = {
    "logistic_regression": {
        "penalty": "l2",
        "C": 0.001,
        "solver": "saga",
        "multi_class": "multinomial",
        "max_iter_per_epoch": 3,
        "tol": 1e-3,
    },
    "random_forest": {
        "rf_tree_step": 5,
        "max_features": "sqrt",
        "n_estimators_start": 0,
        "n_jobs": -1,
    },
    "qda": {
        "reg_param": 1e-3,
    },
    "gaussian_nb": {
        "var_smoothing": 1e-9,
    },
    "knn": {
        "n_neighbors": 5,
    },
}


def get_script_dir():
    """Return the directory containing this script."""

    return os.path.dirname(os.path.abspath(__file__))


def _get_display_names(class_names):
    """Return display abbreviations when available."""

    return [gas_display_name.get(class_name, class_name) for class_name in class_names]


def _get_algorithm_params(
    classifier_name,
    algorithm_config_name=None,
    algorithm_configs=None,
    algorithm_config_overrides=None,
):
    """Load one algorithm's key parameters from the centralized config dict."""

    if algorithm_configs is None:
        algorithm_configs = TRADITIONAL_ML_ALGORITHM_CONFIGS
    config_key = algorithm_config_name or classifier_name
    if config_key not in algorithm_configs:
        raise ValueError(
            "Unsupported algorithm config key: {}. Expected one of {}.".format(
                config_key,
                sorted(algorithm_configs.keys()),
            )
        )

    algorithm_params = dict(algorithm_configs[config_key])
    if algorithm_config_overrides is not None:
        algorithm_params.update(algorithm_config_overrides)
    return algorithm_params


def _load_selected_dataset(
    dataset_name,
    graph_type,
    mydataset_base_data_path,
    generated_base_data_path,
):
    """Load either MyDataset or GeneratedGasImageDataset based on one selector."""

    normalized_dataset_name = dataset_name.lower()
    if normalized_dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            "Unsupported dataset_name: {}. Expected one of {}.".format(
                dataset_name,
                sorted(SUPPORTED_DATASETS),
            )
        )

    if normalized_dataset_name == "mydataset":
        train_dataset = MyDataset(
            mydataset_base_data_path,
            "train",
            transform=None,
            gas_name_array=gas_name_array,
            gas_label=gas_label,
            img_type=graph_type,
        )
        test_dataset = MyDataset(
            mydataset_base_data_path,
            "test",
            transform=None,
            gas_name_array=gas_name_array,
            gas_label=gas_label,
            img_type=graph_type,
        )
        class_names = list(gas_name_array)
        label_names = _get_display_names(class_names)
        dataset_tag = "mydataset"
        return train_dataset, test_dataset, class_names, label_names, dataset_tag

    train_dataset = GeneratedGasImageDataset(
        generated_base_data_path,
        "train",
        transform=None,
    )
    test_dataset = GeneratedGasImageDataset(
        generated_base_data_path,
        "test",
        transform=None,
    )
    class_names = train_dataset.class_names
    label_names = train_dataset.class_display_names
    dataset_tag = "generated"
    return train_dataset, test_dataset, class_names, label_names, dataset_tag


def _build_metric_axes(figure_size, xlabel="epoch", ylabel="Loss"):
    """Create axes styled to match the existing project figures."""

    legend_font = {
        "family": "Arial",
        "style": "normal",
        "size": 30,
        "weight": "bold",
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    bold_font = {"fontname": "Arial", "weight": "bold"}
    axes.spines["bottom"].set_linewidth(5)
    axes.spines["left"].set_linewidth(5)
    axes.spines["right"].set_linewidth(5)
    axes.spines["top"].set_linewidth(5)
    axes.set_xlabel(xlabel, labelpad=1, fontsize=40, **bold_font)
    axes.set_ylabel(ylabel, labelpad=5, fontsize=40, **bold_font)
    axes.tick_params(axis="x", labelsize=36, direction="out", width=4, length=10)
    axes.tick_params(axis="y", labelsize=36, direction="out", width=4, length=10)
    x_label = axes.get_xticklabels()
    [x_label_temp.set_fontweight("bold") for x_label_temp in x_label]
    y_label = axes.get_yticklabels()
    [y_label_temp.set_fontweight("bold") for y_label_temp in y_label]
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    return fig, axes, legend_font


def _save_metric_curve(epoch_list, train_values, test_values, ylabel, save_path, show_plot=True):
    """Save one train/test metric curve figure."""

    fig, axes, legend_font = _build_metric_axes((12, 8), xlabel="epoch", ylabel=ylabel)
    axes.plot(epoch_list, train_values, color="darkblue", label="train {}".format(ylabel.lower()))
    axes.plot(epoch_list, test_values, color="darkred", label="test {}".format(ylabel.lower()))
    axes.legend(prop=legend_font)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def _build_repeated_experiment_axes(figure_size, xlabel="epoch", ylabel="Loss"):
    """Create axes styled to match the repeated-experiment plots in the project."""

    legend_font = {
        "family": "Arial",
        "style": "normal",
        "size": 30,
        "weight": "bold",
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    bold_font = {"fontname": "Arial", "weight": "bold"}
    axes.spines["bottom"].set_linewidth(5)
    axes.spines["left"].set_linewidth(5)
    axes.spines["right"].set_linewidth(5)
    axes.spines["top"].set_linewidth(5)
    axes.set_xlabel(xlabel, labelpad=1, fontsize=40, **bold_font)
    axes.set_ylabel(ylabel, labelpad=5, fontsize=40, **bold_font)
    axes.tick_params(axis="x", labelsize=36, direction="out", width=4, length=10)
    axes.tick_params(axis="y", labelsize=36, direction="out", width=4, length=10)
    x_label = axes.get_xticklabels()
    [x_label_temp.set_fontweight("bold") for x_label_temp in x_label]
    y_label = axes.get_yticklabels()
    [y_label_temp.set_fontweight("bold") for y_label_temp in y_label]
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    return fig, axes, legend_font


def _plot_repeated_experiment_metric(dataframe, train_column, test_column, ylabel, save_path, show_plot=True):
    """Plot light single-trial curves plus one dark mean curve for one metric."""

    fig, axes, legend_font = _build_repeated_experiment_axes((12, 8), xlabel="epoch", ylabel=ylabel)
    run_dataframe = dataframe[dataframe["record_type"] == "trial"].copy()
    run_dataframe["trial_id"] = run_dataframe["trial_id"].astype(str)

    train_pivot = (
        run_dataframe
        .pivot_table(index="epoch", columns="trial_id", values=train_column, aggfunc="mean")
        .sort_index()
    )
    test_pivot = (
        run_dataframe
        .pivot_table(index="epoch", columns="trial_id", values=test_column, aggfunc="mean")
        .sort_index()
    )
    mean_dataframe = dataframe[dataframe["record_type"] == "mean"].copy().sort_values("epoch")

    light_red = "#f3a2a2"
    dark_red = "#8b0000"
    light_blue = "#a8c9ff"
    dark_blue = "#003f8c"

    for trial_id in train_pivot.columns:
        axes.plot(
            train_pivot.index.to_numpy(),
            train_pivot[trial_id].to_numpy(),
            color=light_red,
            linewidth=2.0,
            alpha=0.9,
        )
    for trial_id in test_pivot.columns:
        axes.plot(
            test_pivot.index.to_numpy(),
            test_pivot[trial_id].to_numpy(),
            color=light_blue,
            linewidth=2.0,
            alpha=0.9,
        )

    axes.plot(
        mean_dataframe["epoch"].to_numpy(),
        mean_dataframe[train_column].to_numpy(),
        color=dark_red,
        linewidth=4.0,
        label="train mean",
    )
    axes.plot(
        mean_dataframe["epoch"].to_numpy(),
        mean_dataframe[test_column].to_numpy(),
        color=dark_blue,
        linewidth=4.0,
        label="test mean",
    )
    axes.legend(prop=legend_font)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def draw_confusion_matrix(
    label_true,
    label_pred,
    label_name,
    title="Confusion Matrix",
    pdf_save_path=None,
    dpi=100,
    show_plot=True,
):
    """Plot a normalized confusion matrix using the existing visual style."""

    cm = confusion_matrix(
        y_true=label_true,
        y_pred=label_pred,
        labels=list(range(len(label_name))),
        normalize="true",
    )

    plt.imshow(cm, cmap="Oranges")
    plt.xlabel("Predict label", font={"family": "Arial", "size": 22, "weight": "bold"}, labelpad=10)
    plt.ylabel("Truth label", font={"family": "Arial", "size": 22, "weight": "bold"}, labelpad=15)
    plt.yticks(range(len(label_name)), label_name, fontproperties="Arial", size=16, weight="bold")
    plt.xticks(range(len(label_name)), label_name, fontproperties="Arial", size=16, weight="bold")
    plt.subplots_adjust(left=0.0002, right=1, top=0.95, bottom=0.2)

    cbar = plt.colorbar()
    cbar.mappable.set_clim(0.0, 1.0)
    cbar.ax.tick_params(labelsize=16)
    y_label = cbar.ax.get_yticklabels()
    [y_label_temp.set_fontweight("bold") for y_label_temp in y_label]
    for i in range(len(label_name)):
        for j in range(len(label_name)):
            value = float(format("%.2f" % cm[j, i]))
            color = (1, 1, 1) if value > 0.5 else (0, 0, 0)
            plt.text(i, j, value, verticalalignment="center", horizontalalignment="center", color=color, weight="bold", size=12)

    if pdf_save_path is not None:
        plt.savefig(pdf_save_path, bbox_inches="tight", dpi=dpi)

    plt.tight_layout(pad=0.1)
    if show_plot:
        plt.show()
    plt.close()


def _extract_flatten_image_features(dataset, image_size=(64, 64), color_mode="rgb"):
    """Read image files, resize, flatten, and return feature matrix with labels."""

    feature_list = []
    label_array = np.array(dataset.label_list, dtype=np.int64)
    for image_path in dataset.img_list:
        if not os.path.exists(image_path):
            raise FileNotFoundError("Image file does not exist: {}".format(image_path))
        image = Image.open(image_path)
        if color_mode.lower() == "gray":
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        image = image.resize(image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        feature_list.append(image_array.reshape(-1))
    return np.stack(feature_list, axis=0), label_array


def _prepare_pca_features_from_feature_arrays(train_features, test_features, pca_components=128, random_state=42):
    """Apply StandardScaler and PCA to two already-extracted feature matrices."""

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    max_components = min(
        pca_components,
        train_features_scaled.shape[0],
        train_features_scaled.shape[1],
    )
    if max_components <= 0:
        raise ValueError("pca_components resolved to a non-positive value.")

    pca = PCA(n_components=max_components, random_state=random_state)
    train_features_pca = pca.fit_transform(train_features_scaled)
    test_features_pca = pca.transform(test_features_scaled)
    return train_features_pca, test_features_pca, pca


def _prepare_pca_features(
    train_dataset,
    test_dataset,
    image_size=(64, 64),
    color_mode="rgb",
    pca_components=128,
    random_state=42,
):
    """Extract flattened image features, then apply StandardScaler and PCA."""

    train_features, train_labels = _extract_flatten_image_features(
        dataset=train_dataset,
        image_size=image_size,
        color_mode=color_mode,
    )
    test_features, test_labels = _extract_flatten_image_features(
        dataset=test_dataset,
        image_size=image_size,
        color_mode=color_mode,
    )
    train_features_pca, test_features_pca, pca = _prepare_pca_features_from_feature_arrays(
        train_features=train_features,
        test_features=test_features,
        pca_components=pca_components,
        random_state=random_state,
    )
    return train_features_pca, test_features_pca, train_labels, test_labels, pca


def _predict_probabilities(model, features, classifier_name):
    """Return class probabilities for the supported classifiers."""

    if classifier_name == "logistic_regression":
        return model.predict_proba(features)
    if classifier_name == "random_forest":
        return model.predict_proba(features)
    if classifier_name == "qda":
        return model.predict_proba(features)
    if classifier_name == "gaussian_nb":
        return model.predict_proba(features)
    if classifier_name == "knn":
        return model.predict_proba(features)
    raise ValueError("Unsupported classifier_name: {}".format(classifier_name))


def _sanitize_probabilities(probabilities):
    """Replace invalid probability values and renormalize each row."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")

    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.clip(probabilities, 1e-12, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    invalid_rows = row_sums.squeeze(axis=1) <= 0
    if np.any(invalid_rows):
        probabilities[invalid_rows] = 1.0
        row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / row_sums
    return probabilities


def _ensure_2d_features(feature_matrix):
    """Ensure the input feature representation is two-dimensional."""

    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(-1, 1)
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be a 2D array after conversion.")
    return feature_matrix


def _get_method_specific_cluster_features(model, classifier_name, features):
    """Return the representation used for t-SNE according to the classifier type."""

    if classifier_name == "knn":
        distances, _ = model.kneighbors(features, return_distance=True)
        return _ensure_2d_features(distances)
    if classifier_name == "qda":
        decision_scores = model.decision_function(features)
        return _ensure_2d_features(decision_scores)
    probability_features = _sanitize_probabilities(_predict_probabilities(model, features, classifier_name))
    return _ensure_2d_features(probability_features)


def _evaluate_classifier(model, classifier_name, train_features, train_labels, test_features, test_labels, num_classes):
    """Compute train/test loss and accuracy for one fitted classifier."""

    train_probabilities = _sanitize_probabilities(_predict_probabilities(model, train_features, classifier_name))
    test_probabilities = _sanitize_probabilities(_predict_probabilities(model, test_features, classifier_name))
    train_predictions = np.argmax(train_probabilities, axis=1)
    test_predictions = np.argmax(test_probabilities, axis=1)

    train_loss = log_loss(train_labels, train_probabilities, labels=list(range(num_classes)))
    test_loss = log_loss(test_labels, test_probabilities, labels=list(range(num_classes)))
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    return {
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
    }


def _fit_classifier_with_curves(
    classifier_name,
    train_features,
    train_labels,
    test_features,
    test_labels,
    epoch_num=120,
    random_state=42,
    algorithm_params=None,
):
    """Fit one traditional classifier and collect stage-wise train/test curves."""

    if classifier_name not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            "Unsupported classifier_name: {}. Expected one of {}.".format(
                classifier_name,
                sorted(SUPPORTED_CLASSIFIERS),
            )
        )

    num_classes = len(np.unique(train_labels))
    if algorithm_params is None:
        algorithm_params = _get_algorithm_params(classifier_name)
    epoch_list = []
    train_losses_total = []
    test_losses_total = []
    train_accuracy_total = []
    test_accuracy_total = []
    final_metrics = None

    training_start_time = time.perf_counter()
    if classifier_name == "logistic_regression":
        model = LogisticRegression(
            penalty=algorithm_params.get("penalty", "l2"),
            C=algorithm_params.get("C", 1.0),
            solver=algorithm_params.get("solver", "saga"),
            multi_class=algorithm_params.get("multi_class", "multinomial"),
            warm_start=True,
            max_iter=algorithm_params.get("max_iter_per_epoch", 1),
            tol=algorithm_params.get("tol", 1e-3),
            random_state=random_state,
        )
        for epoch in range(epoch_num):
            model.fit(train_features, train_labels)
            current_metrics = _evaluate_classifier(
                model=model,
                classifier_name=classifier_name,
                train_features=train_features,
                train_labels=train_labels,
                test_features=test_features,
                test_labels=test_labels,
                num_classes=num_classes,
            )
            epoch_list.append(epoch + 1)
            train_losses_total.append(current_metrics["train_loss"])
            test_losses_total.append(current_metrics["test_loss"])
            train_accuracy_total.append(current_metrics["train_accuracy"])
            test_accuracy_total.append(current_metrics["test_accuracy"])
            final_metrics = current_metrics
    elif classifier_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=algorithm_params.get("n_estimators_start", 0),
            warm_start=True,
            random_state=random_state,
            max_features=algorithm_params.get("max_features", "sqrt"),
            n_jobs=algorithm_params.get("n_jobs", -1),
        )
        for epoch in range(epoch_num):
            model.n_estimators += algorithm_params.get("rf_tree_step", 5)
            model.fit(train_features, train_labels)
            current_metrics = _evaluate_classifier(
                model=model,
                classifier_name=classifier_name,
                train_features=train_features,
                train_labels=train_labels,
                test_features=test_features,
                test_labels=test_labels,
                num_classes=num_classes,
            )
            epoch_list.append(epoch + 1)
            train_losses_total.append(current_metrics["train_loss"])
            test_losses_total.append(current_metrics["test_loss"])
            train_accuracy_total.append(current_metrics["train_accuracy"])
            test_accuracy_total.append(current_metrics["test_accuracy"])
            final_metrics = current_metrics
    elif classifier_name == "qda":
        model = QuadraticDiscriminantAnalysis(reg_param=algorithm_params.get("reg_param", 1e-3))
        model.fit(train_features, train_labels)
        final_metrics = _evaluate_classifier(
            model=model,
            classifier_name=classifier_name,
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            num_classes=num_classes,
        )
        epoch_list.append(1)
        train_losses_total.append(final_metrics["train_loss"])
        test_losses_total.append(final_metrics["test_loss"])
        train_accuracy_total.append(final_metrics["train_accuracy"])
        test_accuracy_total.append(final_metrics["test_accuracy"])
    elif classifier_name == "gaussian_nb":
        model = GaussianNB(var_smoothing=algorithm_params.get("var_smoothing", 1e-9))
        model.fit(train_features, train_labels)
        final_metrics = _evaluate_classifier(
            model=model,
            classifier_name=classifier_name,
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            num_classes=num_classes,
        )
        epoch_list.append(1)
        train_losses_total.append(final_metrics["train_loss"])
        test_losses_total.append(final_metrics["test_loss"])
        train_accuracy_total.append(final_metrics["train_accuracy"])
        test_accuracy_total.append(final_metrics["test_accuracy"])
    elif classifier_name == "knn":
        model = KNeighborsClassifier(n_neighbors=algorithm_params.get("n_neighbors", 5))
        model.fit(train_features, train_labels)
        final_metrics = _evaluate_classifier(
            model=model,
            classifier_name=classifier_name,
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            num_classes=num_classes,
        )
        epoch_list.append(1)
        train_losses_total.append(final_metrics["train_loss"])
        test_losses_total.append(final_metrics["test_loss"])
        train_accuracy_total.append(final_metrics["train_accuracy"])
        test_accuracy_total.append(final_metrics["test_accuracy"])
    else:
        raise ValueError("Unsupported classifier_name: {}".format(classifier_name))

    training_time_seconds = time.perf_counter() - training_start_time

    inference_start_time = time.perf_counter()
    _ = model.predict(test_features[:1])
    inference_time_seconds = time.perf_counter() - inference_start_time

    return {
        "model": model,
        "epoch_list": epoch_list,
        "train_losses_total": train_losses_total,
        "test_losses_total": test_losses_total,
        "train_accuracy_total": train_accuracy_total,
        "test_accuracy_total": test_accuracy_total,
        "final_metrics": final_metrics,
        "training_time_seconds": training_time_seconds,
        "inference_time_seconds": inference_time_seconds,
    }


def _save_confusion_matrix_figure(
    label_true,
    label_pred,
    label_names,
    classifier_name,
    graph_type,
    dataset_tag,
    output_dir,
    artifact_tag=None,
    show_plot=True,
):
    """Save the final confusion matrix figure."""

    os.makedirs(output_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_stem = "{}_{}_{}_confusion_matrix".format(classifier_name, dataset_tag, graph_type)
    if artifact_tag:
        file_stem += "_{}".format(artifact_tag)
    file_stem += "_{}".format(time_str)
    save_path = os.path.join(output_dir, file_stem + ".png")
    draw_confusion_matrix(
        label_true=label_true,
        label_pred=label_pred,
        label_name=label_names,
        title="Confusion Matrix",
        pdf_save_path=save_path,
        dpi=300,
        show_plot=show_plot,
    )
    return save_path


def _save_tsne_cluster_figure(
    model,
    classifier_name,
    train_features,
    test_features,
    train_labels,
    test_labels,
    class_names,
    graph_type,
    dataset_tag,
    output_dir,
    artifact_tag=None,
    show_plot=True,
):
    """Run t-SNE on method-specific classifier outputs and save both CSV and scatter plot."""

    os.makedirs(output_dir, exist_ok=True)
    train_cluster_features = _get_method_specific_cluster_features(model, classifier_name, train_features)
    test_cluster_features = _get_method_specific_cluster_features(model, classifier_name, test_features)
    all_features = np.concatenate([train_cluster_features, test_cluster_features], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    split_array = np.array(["train"] * len(train_labels) + ["test"] * len(test_labels))
    perplexity = min(30, max(1, len(all_features) - 1))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    reduced_features = tsne.fit_transform(all_features)

    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_stem = "{}_{}_{}_tsne_cluster".format(classifier_name, dataset_tag, graph_type)
    if artifact_tag:
        file_stem += "_{}".format(artifact_tag)
    file_stem += "_{}".format(time_str)
    csv_path = os.path.join(output_dir, file_stem + ".csv")
    fig_path = os.path.join(output_dir, file_stem + ".png")

    export_rows = []
    for index in range(len(all_features)):
        export_rows.append({
            "split": split_array[index],
            "sample_index": index,
            "true_label": int(all_labels[index]),
            "true_class": class_names[int(all_labels[index])],
            "dim1": float(reduced_features[index, 0]),
            "dim2": float(reduced_features[index, 1]),
        })
    pd.DataFrame(export_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(figsize=(12, 8))
    color_map = plt.cm.get_cmap("tab10", len(class_names))
    for class_index, class_name in enumerate(class_names):
        mask = all_labels == class_index
        if not np.any(mask):
            continue
        axes.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            s=50,
            alpha=0.8,
            color=color_map(class_index),
            label=class_name,
        )
    feature_type_name = {
        "knn": "Neighbor Distance",
        "qda": "Decision Score",
    }.get(classifier_name, "Probability")
    axes.set_title(
        "{} {} {} t-SNE Cluster".format(classifier_name, graph_type, feature_type_name),
        fontsize=16,
        fontweight="bold",
    )
    axes.set_xlabel("t-SNE 1", fontsize=13, fontweight="bold")
    axes.set_ylabel("t-SNE 2", fontsize=13, fontweight="bold")
    axes.grid(alpha=0.25)
    axes.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return csv_path, fig_path


def _save_metric_csv_and_figures(
    classifier_name,
    dataset_tag,
    graph_type,
    epoch_list,
    train_losses_total,
    test_losses_total,
    train_accuracy_total,
    test_accuracy_total,
    pca_components,
    image_size,
    color_mode,
    training_time_seconds,
    inference_time_seconds,
    output_dir,
    show_plot=True,
):
    """Save one metric CSV plus train/test loss and accuracy curves."""

    os.makedirs(output_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_stem = "{}_{}_{}_{}".format(classifier_name, dataset_tag, graph_type, time_str)
    csv_path = os.path.join(output_dir, file_stem + ".csv")
    loss_path = os.path.join(output_dir, file_stem + "_loss.png")
    accuracy_path = os.path.join(output_dir, file_stem + "_accuracy.png")

    dataframe = pd.DataFrame({
        "epoch": epoch_list,
        "train_loss": train_losses_total,
        "test_loss": test_losses_total,
        "train_accuracy": train_accuracy_total,
        "test_accuracy": test_accuracy_total,
        "classifier_name": classifier_name,
        "dataset_name": dataset_tag,
        "graph_type": graph_type,
        "pca_components": pca_components,
        "image_height": image_size[0],
        "image_width": image_size[1],
        "color_mode": color_mode,
        "Training Time (S)": training_time_seconds,
        "Inference Time (S)": inference_time_seconds,
    })
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")

    _save_metric_curve(
        epoch_list,
        train_losses_total,
        test_losses_total,
        ylabel="Loss",
        save_path=loss_path,
        show_plot=show_plot,
    )
    _save_metric_curve(
        epoch_list,
        train_accuracy_total,
        test_accuracy_total,
        ylabel="Accuracy",
        save_path=accuracy_path,
        show_plot=show_plot,
    )
    return csv_path, loss_path, accuracy_path


def _build_trial_epoch_rows(
    trial_id,
    trial_kind,
    classifier_name,
    dataset_tag,
    graph_type,
    epoch_list,
    train_losses_total,
    test_losses_total,
    train_accuracy_total,
    test_accuracy_total,
    pca_components,
    image_size,
    color_mode,
    train_size,
    test_size,
    training_time_seconds,
    inference_time_seconds,
):
    """Build per-epoch rows for one fold/run."""

    rows = []
    for index, epoch in enumerate(epoch_list):
        rows.append({
            "record_type": "trial",
            "trial_kind": trial_kind,
            "trial_id": trial_id,
            "epoch": epoch,
            "classifier_name": classifier_name,
            "dataset_name": dataset_tag,
            "graph_type": graph_type,
            "pca_components": pca_components,
            "image_height": image_size[0],
            "image_width": image_size[1],
            "color_mode": color_mode,
            "train_size": train_size,
            "test_size": test_size,
            "train_loss": train_losses_total[index],
            "test_loss": test_losses_total[index],
            "train_accuracy": train_accuracy_total[index],
            "test_accuracy": test_accuracy_total[index],
            "training_time_seconds": training_time_seconds,
            "inference_time_seconds": inference_time_seconds,
        })
    return rows


def _build_trial_summary_row(
    trial_id,
    trial_kind,
    classifier_name,
    dataset_tag,
    graph_type,
    epoch_num,
    pca_components,
    image_size,
    color_mode,
    fit_result,
    train_size,
    test_size,
):
    """Build one summary row for one fold/run."""

    test_accuracy_array = np.asarray(fit_result["test_accuracy_total"], dtype=np.float64)
    best_epoch_index = int(np.argmax(test_accuracy_array))
    return {
        "trial_kind": trial_kind,
        "trial_id": trial_id,
        "classifier_name": classifier_name,
        "dataset_name": dataset_tag,
        "graph_type": graph_type,
        "epoch_num": epoch_num,
        "pca_components": pca_components,
        "image_height": image_size[0],
        "image_width": image_size[1],
        "color_mode": color_mode,
        "train_size": train_size,
        "test_size": test_size,
        "best_epoch": fit_result["epoch_list"][best_epoch_index],
        "best_train_loss": fit_result["train_losses_total"][best_epoch_index],
        "best_test_loss": fit_result["test_losses_total"][best_epoch_index],
        "best_train_accuracy": fit_result["train_accuracy_total"][best_epoch_index],
        "best_test_accuracy": fit_result["test_accuracy_total"][best_epoch_index],
        "final_train_loss": fit_result["train_losses_total"][-1],
        "final_test_loss": fit_result["test_losses_total"][-1],
        "final_train_accuracy": fit_result["train_accuracy_total"][-1],
        "final_test_accuracy": fit_result["test_accuracy_total"][-1],
        "training_time_seconds": fit_result["training_time_seconds"],
        "inference_time_seconds": fit_result["inference_time_seconds"],
    }


def _save_multi_test_csv_and_figures(
    classifier_name,
    dataset_tag,
    graph_type,
    trial_kind,
    epoch_rows,
    summary_rows,
    output_dir,
    show_plot=True,
):
    """Save multi-test detailed CSV, summary CSV, and mean-curve figures."""

    os.makedirs(output_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_stem = "{}_{}_{}_{}_{}".format(classifier_name, dataset_tag, graph_type, trial_kind, time_str)
    detail_csv_path = os.path.join(output_dir, file_stem + "_detail.csv")
    summary_csv_path = os.path.join(output_dir, file_stem + "_summary.csv")
    loss_path = os.path.join(output_dir, file_stem + "_loss.png")
    accuracy_path = os.path.join(output_dir, file_stem + "_accuracy.png")

    epoch_dataframe = pd.DataFrame(epoch_rows)
    mean_dataframe = (
        epoch_dataframe
        .groupby("epoch", as_index=False)[["train_loss", "test_loss", "train_accuracy", "test_accuracy"]]
        .mean()
    )
    mean_dataframe["record_type"] = "mean"
    mean_dataframe["trial_kind"] = trial_kind
    mean_dataframe["trial_id"] = "mean"
    mean_dataframe["classifier_name"] = classifier_name
    mean_dataframe["dataset_name"] = dataset_tag
    mean_dataframe["graph_type"] = graph_type
    for column_name in [
        "pca_components",
        "image_height",
        "image_width",
        "train_size",
        "test_size",
        "training_time_seconds",
        "inference_time_seconds",
    ]:
        mean_dataframe[column_name] = float(epoch_dataframe[column_name].mean())
    mean_dataframe["color_mode"] = epoch_dataframe["color_mode"].iloc[0]
    detail_dataframe = pd.concat([epoch_dataframe, mean_dataframe], ignore_index=True)
    detail_dataframe.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")

    summary_dataframe = pd.DataFrame(summary_rows)
    metric_columns = [
        "train_size",
        "test_size",
        "best_epoch",
        "best_train_loss",
        "best_test_loss",
        "best_train_accuracy",
        "best_test_accuracy",
        "final_train_loss",
        "final_test_loss",
        "final_train_accuracy",
        "final_test_accuracy",
        "training_time_seconds",
        "inference_time_seconds",
    ]
    summary_mean_row = {
        "trial_kind": trial_kind,
        "trial_id": "mean",
        "classifier_name": classifier_name,
        "dataset_name": dataset_tag,
        "graph_type": graph_type,
        "epoch_num": float(summary_dataframe["epoch_num"].mean()),
        "pca_components": float(summary_dataframe["pca_components"].mean()),
        "image_height": float(summary_dataframe["image_height"].mean()),
        "image_width": float(summary_dataframe["image_width"].mean()),
        "color_mode": summary_dataframe["color_mode"].iloc[0],
    }
    summary_std_row = dict(summary_mean_row)
    summary_std_row["trial_id"] = "std"
    for column_name in metric_columns:
        summary_mean_row[column_name] = float(summary_dataframe[column_name].mean())
        summary_std_row[column_name] = float(summary_dataframe[column_name].std(ddof=0))
    final_summary_dataframe = pd.concat(
        [summary_dataframe, pd.DataFrame([summary_mean_row, summary_std_row])],
        ignore_index=True,
    )
    final_summary_dataframe.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    _plot_repeated_experiment_metric(
        dataframe=detail_dataframe,
        train_column="train_loss",
        test_column="test_loss",
        ylabel="Loss",
        save_path=loss_path,
        show_plot=show_plot,
    )
    _plot_repeated_experiment_metric(
        dataframe=detail_dataframe,
        train_column="train_accuracy",
        test_column="test_accuracy",
        ylabel="Accuracy",
        save_path=accuracy_path,
        show_plot=show_plot,
    )
    return detail_csv_path, summary_csv_path, loss_path, accuracy_path


def train_test_classification_generated_images_traditional_ml(
    classifier_name,
    graph_type,
    dataset_name="generated",
    epoch_num=120,
    pca_components=128,
    image_size=(64, 64),
    color_mode="rgb",
    random_state=42,
    algorithm_config_name=None,
    algorithm_config_overrides=None,
    mydataset_base_data_path=DEFAULT_MYDATASET_BASE_DATA_PATH,
    generated_base_data_path=DEFAULT_GENERATED_BASE_DATA_PATH,
    multi_test=False,
    repeat_times=5,
    mydataset_multi_test_mode="repeat",
    show_plot=True,
):
    """Run one traditional-ML baseline on either MyDataset or GeneratedGasImageDataset."""

    train_dataset, test_dataset, class_names, label_names, dataset_tag = _load_selected_dataset(
        dataset_name=dataset_name,
        graph_type=graph_type,
        mydataset_base_data_path=mydataset_base_data_path,
        generated_base_data_path=generated_base_data_path,
    )
    algorithm_params = _get_algorithm_params(
        classifier_name=classifier_name,
        algorithm_config_name=algorithm_config_name,
        algorithm_config_overrides=algorithm_config_overrides,
    )

    if multi_test:
        if repeat_times != 5:
            raise ValueError("multi_test mode is fixed to five repeated validations, so repeat_times must be 5.")
        normalized_mydataset_multi_test_mode = mydataset_multi_test_mode.lower()
        if normalized_mydataset_multi_test_mode not in {"repeat", "cross_validation"}:
            raise ValueError(
                "Unsupported mydataset_multi_test_mode: {}. Expected 'repeat' or 'cross_validation'.".format(
                    mydataset_multi_test_mode
                )
            )

        results_dir = os.path.join(get_script_dir(), RESULTS_MULTI_DIR_NAME)
        confusion_dir = os.path.join(get_script_dir(), CONFUSION_MULTI_DIR_NAME)
        cluster_dir = os.path.join(get_script_dir(), CLUSTER_MULTI_DIR_NAME)
        epoch_rows = []
        summary_rows = []
        if dataset_tag == "mydataset" and normalized_mydataset_multi_test_mode == "cross_validation":
            train_features_raw, train_labels = _extract_flatten_image_features(
                train_dataset,
                image_size=image_size,
                color_mode=color_mode,
            )
            test_features_raw, test_labels = _extract_flatten_image_features(
                test_dataset,
                image_size=image_size,
                color_mode=color_mode,
            )
            full_features_raw = np.concatenate([train_features_raw, test_features_raw], axis=0)
            full_labels = np.concatenate([train_labels, test_labels], axis=0)
            sample_index_array = np.arange(len(full_labels))
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

            for fold_index, (train_index, val_index) in enumerate(
                splitter.split(sample_index_array, full_labels),
                start=1,
            ):
                fold_seed = random_state + fold_index - 1
                train_features, test_features, pca = _prepare_pca_features_from_feature_arrays(
                    train_features=full_features_raw[train_index],
                    test_features=full_features_raw[val_index],
                    pca_components=pca_components,
                    random_state=fold_seed,
                )
                fold_train_labels = full_labels[train_index]
                fold_test_labels = full_labels[val_index]
                fit_result = _fit_classifier_with_curves(
                    classifier_name=classifier_name,
                    train_features=train_features,
                    train_labels=fold_train_labels,
                    test_features=test_features,
                    test_labels=fold_test_labels,
                    epoch_num=epoch_num,
                    random_state=fold_seed,
                    algorithm_params=algorithm_params,
                )
                _save_confusion_matrix_figure(
                    label_true=fold_test_labels.tolist(),
                    label_pred=fit_result["final_metrics"]["test_predictions"].tolist(),
                    label_names=label_names,
                    classifier_name=classifier_name,
                    graph_type=graph_type,
                    dataset_tag=dataset_tag,
                    output_dir=confusion_dir,
                    artifact_tag="fold{}".format(fold_index),
                    show_plot=show_plot,
                )
                _save_tsne_cluster_figure(
                    model=fit_result["model"],
                    classifier_name=classifier_name,
                    train_features=train_features,
                    test_features=test_features,
                    train_labels=fold_train_labels,
                    test_labels=fold_test_labels,
                    class_names=class_names,
                    graph_type=graph_type,
                    dataset_tag=dataset_tag,
                    output_dir=cluster_dir,
                    artifact_tag="fold{}".format(fold_index),
                    show_plot=show_plot,
                )
                epoch_rows.extend(
                    _build_trial_epoch_rows(
                        trial_id=fold_index,
                        trial_kind="fold",
                        classifier_name=classifier_name,
                        dataset_tag=dataset_tag,
                        graph_type=graph_type,
                        epoch_list=fit_result["epoch_list"],
                        train_losses_total=fit_result["train_losses_total"],
                        test_losses_total=fit_result["test_losses_total"],
                        train_accuracy_total=fit_result["train_accuracy_total"],
                        test_accuracy_total=fit_result["test_accuracy_total"],
                        pca_components=pca.n_components_,
                        image_size=image_size,
                        color_mode=color_mode,
                        train_size=len(train_index),
                        test_size=len(val_index),
                        training_time_seconds=fit_result["training_time_seconds"],
                        inference_time_seconds=fit_result["inference_time_seconds"],
                    )
                )
                summary_rows.append(
                    _build_trial_summary_row(
                        trial_id=fold_index,
                        trial_kind="fold",
                        classifier_name=classifier_name,
                        dataset_tag=dataset_tag,
                        graph_type=graph_type,
                        epoch_num=epoch_num,
                        pca_components=pca.n_components_,
                        image_size=image_size,
                        color_mode=color_mode,
                        fit_result=fit_result,
                        train_size=len(train_index),
                        test_size=len(val_index),
                    )
                )
            detail_csv_path, summary_csv_path, loss_path, accuracy_path = _save_multi_test_csv_and_figures(
                classifier_name=classifier_name,
                dataset_tag=dataset_tag,
                graph_type=graph_type,
                trial_kind="five_fold",
                epoch_rows=epoch_rows,
                summary_rows=summary_rows,
                output_dir=results_dir,
                show_plot=show_plot,
            )
        else:
            train_features_raw, train_labels = _extract_flatten_image_features(
                train_dataset,
                image_size=image_size,
                color_mode=color_mode,
            )
            test_features_raw, test_labels = _extract_flatten_image_features(
                test_dataset,
                image_size=image_size,
                color_mode=color_mode,
            )
            for run_index in range(1, repeat_times + 1):
                run_seed = random_state + run_index - 1
                train_features, test_features, pca = _prepare_pca_features_from_feature_arrays(
                    train_features=train_features_raw,
                    test_features=test_features_raw,
                    pca_components=pca_components,
                    random_state=run_seed,
                )
                fit_result = _fit_classifier_with_curves(
                    classifier_name=classifier_name,
                    train_features=train_features,
                    train_labels=train_labels,
                    test_features=test_features,
                    test_labels=test_labels,
                    epoch_num=epoch_num,
                    random_state=run_seed,
                    algorithm_params=algorithm_params,
                )
                _save_confusion_matrix_figure(
                    label_true=test_labels.tolist(),
                    label_pred=fit_result["final_metrics"]["test_predictions"].tolist(),
                    label_names=label_names,
                    classifier_name=classifier_name,
                    graph_type=graph_type,
                    dataset_tag=dataset_tag,
                    output_dir=confusion_dir,
                    artifact_tag="run{}".format(run_index),
                    show_plot=show_plot,
                )
                _save_tsne_cluster_figure(
                    model=fit_result["model"],
                    classifier_name=classifier_name,
                    train_features=train_features,
                    test_features=test_features,
                    train_labels=train_labels,
                    test_labels=test_labels,
                    class_names=class_names,
                    graph_type=graph_type,
                    dataset_tag=dataset_tag,
                    output_dir=cluster_dir,
                    artifact_tag="run{}".format(run_index),
                    show_plot=show_plot,
                )
                epoch_rows.extend(
                    _build_trial_epoch_rows(
                        trial_id=run_index,
                        trial_kind="run",
                        classifier_name=classifier_name,
                        dataset_tag=dataset_tag,
                        graph_type=graph_type,
                        epoch_list=fit_result["epoch_list"],
                        train_losses_total=fit_result["train_losses_total"],
                        test_losses_total=fit_result["test_losses_total"],
                        train_accuracy_total=fit_result["train_accuracy_total"],
                        test_accuracy_total=fit_result["test_accuracy_total"],
                        pca_components=pca.n_components_,
                        image_size=image_size,
                        color_mode=color_mode,
                        train_size=len(train_labels),
                        test_size=len(test_labels),
                        training_time_seconds=fit_result["training_time_seconds"],
                        inference_time_seconds=fit_result["inference_time_seconds"],
                    )
                )
                summary_rows.append(
                    _build_trial_summary_row(
                        trial_id=run_index,
                        trial_kind="run",
                        classifier_name=classifier_name,
                        dataset_tag=dataset_tag,
                        graph_type=graph_type,
                        epoch_num=epoch_num,
                        pca_components=pca.n_components_,
                        image_size=image_size,
                        color_mode=color_mode,
                        fit_result=fit_result,
                        train_size=len(train_labels),
                        test_size=len(test_labels),
                    )
                )
            detail_csv_path, summary_csv_path, loss_path, accuracy_path = _save_multi_test_csv_and_figures(
                classifier_name=classifier_name,
                dataset_tag=dataset_tag,
                graph_type=graph_type,
                trial_kind="repeat",
                epoch_rows=epoch_rows,
                summary_rows=summary_rows,
                output_dir=results_dir,
                show_plot=show_plot,
            )

        print("[TraditionalML] Saved multi-test detail CSV to: {}".format(detail_csv_path))
        print("[TraditionalML] Saved multi-test summary CSV to: {}".format(summary_csv_path))
        print("[TraditionalML] Saved multi-test loss curve to: {}".format(loss_path))
        print("[TraditionalML] Saved multi-test accuracy curve to: {}".format(accuracy_path))
        return {
            "detail_csv_path": detail_csv_path,
            "summary_csv_path": summary_csv_path,
            "loss_path": loss_path,
            "accuracy_path": accuracy_path,
        }

    train_features, test_features, train_labels, test_labels, pca = _prepare_pca_features(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        image_size=image_size,
        color_mode=color_mode,
        pca_components=pca_components,
        random_state=random_state,
    )
    fit_result = _fit_classifier_with_curves(
        classifier_name=classifier_name,
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        epoch_num=epoch_num,
        random_state=random_state,
        algorithm_params=algorithm_params,
    )

    results_dir = os.path.join(get_script_dir(), RESULTS_DIR_NAME)
    confusion_dir = os.path.join(get_script_dir(), CONFUSION_DIR_NAME)
    cluster_dir = os.path.join(get_script_dir(), CLUSTER_DIR_NAME)

    csv_path, loss_path, accuracy_path = _save_metric_csv_and_figures(
        classifier_name=classifier_name,
        dataset_tag=dataset_tag,
        graph_type=graph_type,
        epoch_list=fit_result["epoch_list"],
        train_losses_total=fit_result["train_losses_total"],
        test_losses_total=fit_result["test_losses_total"],
        train_accuracy_total=fit_result["train_accuracy_total"],
        test_accuracy_total=fit_result["test_accuracy_total"],
        pca_components=pca.n_components_,
        image_size=image_size,
        color_mode=color_mode,
        training_time_seconds=fit_result["training_time_seconds"],
        inference_time_seconds=fit_result["inference_time_seconds"],
        output_dir=results_dir,
        show_plot=show_plot,
    )
    confusion_path = _save_confusion_matrix_figure(
        label_true=test_labels.tolist(),
        label_pred=fit_result["final_metrics"]["test_predictions"].tolist(),
        label_names=label_names,
        classifier_name=classifier_name,
        graph_type=graph_type,
        dataset_tag=dataset_tag,
        output_dir=confusion_dir,
        show_plot=show_plot,
    )
    cluster_csv_path, cluster_fig_path = _save_tsne_cluster_figure(
        model=fit_result["model"],
        classifier_name=classifier_name,
        train_features=train_features,
        test_features=test_features,
        train_labels=train_labels,
        test_labels=test_labels,
        class_names=class_names,
        graph_type=graph_type,
        dataset_tag=dataset_tag,
        output_dir=cluster_dir,
        show_plot=show_plot,
    )

    print("[TraditionalML] Saved metric CSV to: {}".format(csv_path))
    print("[TraditionalML] Saved loss curve to: {}".format(loss_path))
    print("[TraditionalML] Saved accuracy curve to: {}".format(accuracy_path))
    print("[TraditionalML] Saved confusion matrix to: {}".format(confusion_path))
    print("[TraditionalML] Saved t-SNE CSV to: {}".format(cluster_csv_path))
    print("[TraditionalML] Saved t-SNE figure to: {}".format(cluster_fig_path))
    return {
        "metric_csv_path": csv_path,
        "loss_path": loss_path,
        "accuracy_path": accuracy_path,
        "confusion_path": confusion_path,
        "cluster_csv_path": cluster_csv_path,
        "cluster_fig_path": cluster_fig_path,
    }


if __name__ == "__main__":
    # train_test_classification_generated_images_traditional_ml(
    #     classifier_name=model_list[3],
    #     graph_type="GASF_img",
    #     dataset_name="mydataset",
    #     epoch_num=120,
    #     pca_components=128,
    #     image_size=(64, 64),
    #     color_mode="rgb",
    #     multi_test= True,
    #     mydataset_multi_test_mode="cross_validation",
    #     random_state=42,
    #     show_plot=True,
    # )

    # train_test_classification_generated_images_traditional_ml(
    #     classifier_name=model_list[3],
    #     graph_type="generated_gasf",
    #     dataset_name="generated",
    #     epoch_num=120,
    #     pca_components=128,
    #     image_size=(64, 64),
    #     multi_test=True,
    #     color_mode="rgb",
    #     random_state=42,
    #     show_plot=True,
    # )
