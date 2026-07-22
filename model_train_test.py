from model import (
    build_classification_ablation_model,
    build_classification_alexnet_multiscale_model,
    build_classification_hybrid_alexnet_multiscale_model,
    build_classification_plain_cnn_model,
    build_classification_stable_alexnet_multiscale_model,
    build_regression_ablation_model,
    build_regression_plain_cnn_model,
)
from model2 import build_regression_model
from model3 import (
    build_fastvit_t8_classification_model,
    build_fastvit_t8_regression_model,
    build_ghostnetv3_100_classification_model,
    build_ghostnetv3_100_regression_model,
    build_starnet_s2_classification_model,
    build_starnet_s2_regression_model,
)
import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from Dataset import GeneratedGasImageDataset, MyDataset, MyDataset_regression
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_curve, auc, PrecisionRecallDisplay, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from draw_fig import *
from torchvision import transforms
from torchvision import models
import datetime
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties

img = ["GASF_img", "CWT_img", "MTF_img", "STFT_img"]
grad_block = []
feature_block = []
fc_features = {
        'fc_input': None,
        'fc_output': None
    }

base_data_path = r"D:\E-nose_DataSet\Gas_Recognition_DataSet\gas_response_data"
generated_img_base_data_path = r"D:\E-nose_DataSet\Gas_Recognition_DataSet\gas_response_data2"
SMALL_GAS_NAME_ARRAY = ["1-butanol", "acetone", "benzaldehyde", "butyl acetate", "dimethylbenzene"]
SMALL_GAS_LABEL = {"1-butanol": 0, "acetone": 1, "benzaldehyde": 2, "butyl acetate": 3, "dimethylbenzene": 4}
SMALL_GAS_DISPLAY_ARRAY = ["1BA", "AC", "BZ", "BAC", "DMB"]
LARGE_GAS_NAME_ARRAY = [
    "1-butanol",
    "acetone",
    "benzaldehyde",
    "butyl acetate",
    "dimethylbenzene",
    "ethanol",
    "ethyl acetate",
    "isopropanol",
    "methanol",
    "methylbenzene",
    "n-heptane",
    "n-propanol",
]
LARGE_GAS_LABEL = {
    "1-butanol": 0,
    "acetone": 1,
    "benzaldehyde": 2,
    "butyl acetate": 3,
    "dimethylbenzene": 4,
    "ethanol": 5,
    "ethyl acetate": 6,
    "isopropanol": 7,
    "methanol": 8,
    "methylbenzene": 9,
    "n-heptane": 10,
    "n-propanol": 11,
}
LARGE_GAS_DISPLAY_ARRAY = [
    "1BA",
    "AC",
    "BZ",
    "BAC",
    "DMB",
    "EA",
    "EAC",
    "IPA",
    "MT",
    "MB",
    "NHE",
    "NPA",
]

gas_name_array = SMALL_GAS_NAME_ARRAY
gas_label = SMALL_GAS_LABEL
generated_gas_name_array = ["acetone", "benzaldehyde", "butyl acetate", "dimethylbenzene", "methanol"]
gas_name_display_array = SMALL_GAS_DISPLAY_ARRAY
generated_gas_name_display_array = ["AC", "BZ", "BAC", "DMB", "MT"]
gas_name_display_map = {
    "1-butanol": "1BA",
    "acetone": "AC",
    "benzaldehyde": "BZ",
    "butyl acetate": "BAC",
    "dimethylbenzene": "DMB",
    "ethanol": "EA",
    "ethyl acetate": "EAC",
    "isopropanol": "IPA",
    "methanol": "MT",
    "methylbenzene": "MB",
    "n-heptane": "NHE",
    "n-propanol": "NPA",
}

model_list=[
    "mine_all_components",
    "mine_multiscale_attention",
    "mine_multiscale_residual",
    "mine_attention_residual",

    "mine_multiscale_only",

    "mine_attention_only",
    "mine_residual_only",

    "mine_plain_cnn",
    "mine_alexnet_multiscale",
    "mine_alexnet_multiscale_stable",
    "mine_alexnet_multiscale_hybrid",

    "resnet50",
    "vgg19",
    "mobilenet_v3_large",
    "AlexNet",
    "GhostNetV3-1.0",
    "StarNet-S2",
    "FastViT-T8",
]
REGRESSION_PPM_MIN = 10.0
REGRESSION_PPM_MAX = 100.0
CLASSIFICATION_ABLATION_CONFIGS = {
    "mine_all_components": (True, True, True),
    "mine_multiscale_attention": (True, True, False),
    "mine_multiscale_residual": (True, False, True),
    "mine_attention_residual": (False, True, True),
    "mine_multiscale_only": (True, False, False),
    "mine_attention_only": (False, True, False),
    "mine_residual_only": (False, False, True),
}
REGRESSION_ABLATION_CONFIGS = CLASSIFICATION_ABLATION_CONFIGS
TIMM_CLASSIFICATION_BUILDERS = {
    "GhostNetV3-1.0": build_ghostnetv3_100_classification_model,
    "ghostnetv3_100": build_ghostnetv3_100_classification_model,
    "StarNet-S2": build_starnet_s2_classification_model,
    "starnet_s2": build_starnet_s2_classification_model,
    "FastViT-T8": build_fastvit_t8_classification_model,
    "fastvit_t8": build_fastvit_t8_classification_model,
}
TIMM_REGRESSION_BUILDERS = {
    "GhostNetV3-1.0": build_ghostnetv3_100_regression_model,
    "ghostnetv3_100": build_ghostnetv3_100_regression_model,
    "StarNet-S2": build_starnet_s2_regression_model,
    "starnet_s2": build_starnet_s2_regression_model,
    "FastViT-T8": build_fastvit_t8_regression_model,
    "fastvit_t8": build_fastvit_t8_regression_model,
}
# Default to random initialization so local training does not block on remote weight downloads.
DEFAULT_USE_PRETRAINED = os.getenv("GAS_USE_PRETRAINED", "0").strip().lower() in {"1", "true", "yes", "y"}

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize generated graph images to the model input size.
    transforms.ToTensor(),
    transforms.Normalize(0.485, 0.229)
    #transforms.Normalize((0.485,0.456, 0.406), (0.229, 0.224, 0.225))

])


def backward_hook(module, grad_in, grad_out):
    """Store backward gradients for CAM-style visualization."""

    grad_block.append(grad_out[0].detach())

def forward_hook(module, input, output):
    """Store forward feature maps for CAM-style visualization."""

    feature_block.append(output)


def forward_hook_function(module, input, output):
    """Capture fully connected layer inputs and outputs for analysis."""

    fc_features['fc_input'] = input[0].clone().detach()
    fc_features['fc_output'] = output.clone().detach()


def get_display_names_for_classes(class_names):
    """Return short display labels when known, otherwise keep the original names."""

    return [gas_name_display_map.get(class_name, class_name) for class_name in class_names]


def get_dataset_one_classification_config(is_large_test=False):
    """Return dataset-one class names, labels, and display labels for 5-class or 12-class experiments."""

    if is_large_test:
        return LARGE_GAS_NAME_ARRAY, LARGE_GAS_LABEL, LARGE_GAS_DISPLAY_ARRAY
    return SMALL_GAS_NAME_ARRAY, SMALL_GAS_LABEL, SMALL_GAS_DISPLAY_ARRAY


def resolve_confusion_matrix_label_names(class_names):
    """Return dataset-specific confusion-matrix display labels in the expected order."""

    class_names = list(class_names)
    if class_names == SMALL_GAS_NAME_ARRAY:
        return SMALL_GAS_DISPLAY_ARRAY
    if class_names == LARGE_GAS_NAME_ARRAY:
        return LARGE_GAS_DISPLAY_ARRAY
    if class_names == generated_gas_name_array:
        return generated_gas_name_display_array
    return get_display_names_for_classes(class_names)


def get_script_dir():
    """Return the directory containing this training script."""

    return os.path.dirname(os.path.abspath(__file__))


def get_last_linear_layer(model):
    """Return the last ``nn.Linear`` layer in ``model`` for feature extraction."""

    last_name = None
    last_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_layer = module
    if last_layer is None:
        raise ValueError("The loaded model does not contain an nn.Linear layer.")
    return last_name, last_layer


def collect_last_fc_features(model, dataset, batch_size, device, split_name, class_names=None):
    """Collect features before the final fully connected layer for one dataset split."""

    if class_names is None:
        class_names = getattr(dataset, "class_names", gas_name_array)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    _, last_linear = get_last_linear_layer(model)
    collected_inputs = []

    def hook_fn(module, module_input, module_output):
        collected_inputs.append(module_input[0].detach().cpu())

    handle = last_linear.register_forward_hook(hook_fn)
    rows = []
    sample_offset = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            collected_inputs.clear()
            outputs = model(inputs.to(device))
            if not collected_inputs:
                raise RuntimeError("Failed to capture the last fully connected layer features.")
            batch_features = collected_inputs[-1].numpy()
            batch_predictions = outputs.argmax(dim=1).detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy()
            batch_size_actual = len(batch_labels)
            batch_paths = dataset.img_list[sample_offset: sample_offset + batch_size_actual]
            for index in range(batch_size_actual):
                rows.append({
                    "split": split_name,
                    "sample_index": sample_offset + index,
                    "file_path": batch_paths[index],
                    "true_label": int(batch_labels[index]),
                    "true_class": class_names[int(batch_labels[index])],
                    "pred_label": int(batch_predictions[index]),
                    "pred_class": class_names[int(batch_predictions[index])],
                    "feature_vector": batch_features[index],
                })
            sample_offset += batch_size_actual
    handle.remove()
    return rows


def reduce_fc_features(feature_matrix, method="pca"):
    """Reduce final fully connected features to 2D using PCA, t-SNE, or UMAP."""

    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(feature_matrix)
        axis_labels = (
            f"PCA 1 ({reducer.explained_variance_ratio_[0] * 100:.2f}%)",
            f"PCA 2 ({reducer.explained_variance_ratio_[1] * 100:.2f}%)",
        )
        return reduced_features, axis_labels
    if method == "tsne":
        sample_count = feature_matrix.shape[0]
        perplexity = min(30, max(1, sample_count - 1))
        reducer = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        reduced_features = reducer.fit_transform(feature_matrix)
        return reduced_features, ("t-SNE 1", "t-SNE 2")
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP requires the 'umap-learn' package. Please install it in the current Python environment."
            ) from exc
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(feature_matrix)
        return reduced_features, ("UMAP 1", "UMAP 2")
    raise ValueError(f"Unknown dimensionality reduction method: {method}")


def export_last_fc_clusters(
    model,
    model_name,
    graph_type,
    batch_size,
    train_dataset,
    test_dataset,
    device,
    reduce_method="pca",
    class_names=None,
    output_dir_name="cluster",
    artifact_tag=None,
):
    """Reduce final fully connected features, cluster them, then save CSV and scatter plot."""

    if class_names is None:
        class_names = getattr(train_dataset, "class_names", gas_name_array)

    train_rows = collect_last_fc_features(model, train_dataset, batch_size, device, "train", class_names=class_names)
    test_rows = collect_last_fc_features(model, test_dataset, batch_size, device, "test", class_names=class_names)
    all_rows = train_rows + test_rows
    if len(all_rows) < 2:
        raise ValueError("At least two samples are required to run feature clustering.")

    feature_matrix = np.stack([row["feature_vector"] for row in all_rows], axis=0)
    reduced_features, axis_labels = reduce_fc_features(feature_matrix, method=reduce_method)

    cluster_count = len(np.unique([row["true_label"] for row in all_rows]))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_features)

    export_rows = []
    for index, row in enumerate(all_rows):
        export_rows.append({
            "split": row["split"],
            "sample_index": row["sample_index"],
            "file_path": row["file_path"],
            "reduce_method": reduce_method.lower(),
            "true_label": row["true_label"],
            "true_class": row["true_class"],
            "pred_label": row["pred_label"],
            "pred_class": row["pred_class"],
            "cluster_label": int(cluster_labels[index]),
            "dim1": float(reduced_features[index, 0]),
            "dim2": float(reduced_features[index, 1]),
        })

    cluster_dir = os.path.join(get_script_dir(), output_dir_name)
    os.makedirs(cluster_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_stem = f"{model_name}_{graph_type}_{reduce_method.lower()}_cluster"
    if artifact_tag:
        file_stem += f"_{artifact_tag}"
    file_stem += f"_{time_str}"
    csv_path = os.path.join(cluster_dir, file_stem + ".csv")
    fig_path = os.path.join(cluster_dir, file_stem + ".png")

    pd.DataFrame(export_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 8))
    color_map = plt.cm.get_cmap("tab10", len(class_names))
    for class_index, class_name in enumerate(class_names):
        mask = np.array([row["true_label"] == class_index for row in all_rows])
        if not np.any(mask):
            continue
        ax.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            s=50,
            alpha=0.8,
            color=color_map(class_index),
            label=class_name,
        )
    method_name = reduce_method.upper() if reduce_method.lower() != "tsne" else "t-SNE"
    ax.set_title(f"{model_name} {graph_type} Last FC Feature {method_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel(axis_labels[0], fontsize=13, fontweight="bold")
    ax.set_ylabel(axis_labels[1], fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    linear_name, _ = get_last_linear_layer(model)
    print(f"[Cluster] Last FC layer: {linear_name}")
    print(f"[Cluster] Reduction method: {reduce_method.lower()}")
    print(f"[Cluster] Saved cluster CSV to: {csv_path}")
    print(f"[Cluster] Saved cluster figure to: {fig_path}")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        """Initialize the meter state."""

        self.reset()

    def reset(self):
        """Reset current value, accumulated sum, count, and average."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Add ``n`` observations with value ``val`` and update the average."""

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_model_param_count(model, model_name="model"):
    """Print total and trainable parameter counts for one model."""

    if model is None:
        print(f"[ParamCount] {model_name}: model is None, cannot count parameters.")
        return
    total_params, trainable_params = count_model_parameters(model)
    print(f"[ParamCount] {model_name} total params: {total_params:,}, trainable params: {trainable_params:,}")

def count_model_parameters(model):
    """Return total and trainable parameter counts for one model."""

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_large_count(value):
    """Format a large numeric count into K/M/G units for tables."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    value = float(value)
    if value >= 1e9:
        return f"{value / 1e9:.4f}G"
    if value >= 1e6:
        return f"{value / 1e6:.4f}M"
    if value >= 1e3:
        return f"{value / 1e3:.4f}K"
    return f"{value:.0f}"


def print_selected_model_param_counts(
    target_model_list,
    in_channels=3,
    use_pretrained=False,
):
    """Build selected models and print their parameter counts in compact units."""

    if target_model_list is None:
        raise ValueError("target_model_list cannot be None.")
    for current_model_name in target_model_list:
        model = load_model(
            current_model_name,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
            verbose=False,
        )
        total_params, trainable_params = count_model_parameters(model)
        print(
            f"[ParamCount] {current_model_name}: "
            f"total={format_large_count(total_params)}, "
            f"trainable={format_large_count(trainable_params)}"
        )


def _build_torchvision_model(model_name, use_pretrained=DEFAULT_USE_PRETRAINED):
    """Create one torchvision model while supporting old and new weight APIs."""

    model_name_to_constructor = {
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "vgg19": "vgg19",
        "AlexNet": "alexnet",
        "mobilenet_v3_large": "mobilenet_v3_large",
        "googleNet": "googlenet",
    }
    constructor_name = model_name_to_constructor.get(model_name)
    if constructor_name is None:
        raise ValueError(f"Unsupported torchvision model_name: {model_name}")

    constructor = getattr(models, constructor_name)
    weights_enum_name_map = {
        "resnet18": "ResNet18_Weights",
        "resnet50": "ResNet50_Weights",
        "vgg19": "VGG19_Weights",
        "AlexNet": "AlexNet_Weights",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
        "googleNet": "GoogLeNet_Weights",
    }
    weights_enum = getattr(models, weights_enum_name_map[model_name], None)

    if use_pretrained:
        if weights_enum is not None:
            return constructor(weights=weights_enum.DEFAULT)
        return constructor(pretrained=True)
    try:
        return constructor(weights=None)
    except TypeError:
        return constructor(pretrained=False)


def profile_model_complexity(model, input_size=(1, 3, 64, 64), device=None):
    """
    Profile one model and return params, MACs and estimated FLOPs.

    FLOPs are estimated as ``2 x MACs``. This convention should be reported in
    any paper or figure that uses the exported table.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    total_params, trainable_params = count_model_parameters(model)
    macs = None
    flops = None
    profiler_name = None

    with torch.no_grad():
        sample_input = torch.randn(*input_size, device=device)
        try:
            from thop import profile
            macs, _ = profile(model, inputs=(sample_input,), verbose=False)
            profiler_name = "thop"
        except ImportError:
            try:
                from ptflops import get_model_complexity_info
                macs, _ = get_model_complexity_info(
                    model,
                    tuple(input_size[1:]),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )
                profiler_name = "ptflops"
            except ImportError as exc:
                raise ImportError(
                    "FLOPs profiling requires 'thop' or 'ptflops'. Please install one of them "
                    "in the current Python environment."
                ) from exc

    if macs is not None:
        macs = float(macs)
        flops = macs * 2.0
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "macs": macs,
        "flops": flops,
        "profiler": profiler_name,
    }


def analyze_model_list_complexity(
    target_model_list=None,
    input_size=(1, 3, 64, 64),
    in_channels=3,
    use_pretrained=False,
    save_dir='GasRecognition',
    sort_by="flops",
):
    """Analyze parameter count and FLOPs for all classification models and save a CSV table."""

    if target_model_list is None:
        target_model_list = model_list
    if save_dir is None:
        save_dir = os.path.join(get_script_dir(), "model_complexity_results")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for current_model_name in target_model_list:
        print(f"[Complexity] analyzing {current_model_name} ...")
        try:
            model = load_model(
                current_model_name,
                in_channels=in_channels,
                use_pretrained=use_pretrained,
                verbose=False,
            )
            metrics = profile_model_complexity(model, input_size=input_size, device=device)
            rows.append({
                "model_name": current_model_name,
                "is_ours": current_model_name.startswith("mine_"),
                "input_size": "x".join(str(dim) for dim in input_size),
                "total_params": metrics["total_params"],
                "trainable_params": metrics["trainable_params"],
                "total_params_fmt": format_large_count(metrics["total_params"]),
                "trainable_params_fmt": format_large_count(metrics["trainable_params"]),
                "macs": metrics["macs"],
                "macs_fmt": format_large_count(metrics["macs"]),
                "flops": metrics["flops"],
                "flops_fmt": format_large_count(metrics["flops"]),
                "profiler": metrics["profiler"],
                "status": "ok",
                "error_message": "",
            })
        except Exception as exc:
            rows.append({
                "model_name": current_model_name,
                "is_ours": current_model_name.startswith("mine_"),
                "input_size": "x".join(str(dim) for dim in input_size),
                "total_params": np.nan,
                "trainable_params": np.nan,
                "total_params_fmt": "",
                "trainable_params_fmt": "",
                "macs": np.nan,
                "macs_fmt": "",
                "flops": np.nan,
                "flops_fmt": "",
                "profiler": "",
                "status": "failed",
                "error_message": repr(exc),
            })

    dataframe = pd.DataFrame(rows)
    if sort_by in dataframe.columns:
        dataframe = dataframe.sort_values(by=sort_by, ascending=True, na_position="last").reset_index(drop=True)

    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_path = os.path.join(save_dir, f"classification_model_complexity_{time_str}.csv")
    dataframe.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[Complexity] Saved model complexity CSV to: {save_path}")
    return dataframe, save_path


def load_model(model_name, in_channels=3, use_pretrained=DEFAULT_USE_PRETRAINED, verbose=True, num_classes=5):
    """Build a classification model by name, including ablation variants."""

    model = None
    timm_builder = TIMM_CLASSIFICATION_BUILDERS.get(model_name)
    if timm_builder is not None:
        model = timm_builder(
            num_classes=num_classes,
            in_channels=in_channels,
            use_pretrained=use_pretrained,
        )
    if model_name in CLASSIFICATION_ABLATION_CONFIGS:
        use_multiscale, use_attention, use_residual = CLASSIFICATION_ABLATION_CONFIGS[model_name]
        model = build_classification_ablation_model(
            num_classes=num_classes,
            in_channels=in_channels,
            use_multiscale=use_multiscale,
            use_attention=use_attention,
            use_residual=use_residual,
        )
    if model_name == "mine_plain_cnn":
        model = build_classification_plain_cnn_model(num_classes, in_channels=in_channels)
    if model_name == "mine_alexnet_multiscale":
        model = build_classification_alexnet_multiscale_model(num_classes, in_channels=in_channels)
    if model_name == "mine_alexnet_multiscale_stable":
        model = build_classification_stable_alexnet_multiscale_model(num_classes, in_channels=in_channels)
    if model_name == "mine_alexnet_multiscale_hybrid":
        model = build_classification_hybrid_alexnet_multiscale_model(num_classes, in_channels=in_channels)
    if model_name == "resnet18":
        model = _build_torchvision_model("resnet18", use_pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Softmax(dim=1))
    if model_name == "resnet50":
        model = _build_torchvision_model("resnet50", use_pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Softmax(dim=1))
    if model_name == "vgg19":
        model = _build_torchvision_model("vgg19", use_pretrained=use_pretrained)
        model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, num_classes), nn.Softmax(dim=1))
    if model_name == "AlexNet":
        model = _build_torchvision_model("AlexNet", use_pretrained=use_pretrained)
        model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, num_classes), nn.Softmax(dim=1))
    if model_name == "mobilenet_v3_large":
        model = _build_torchvision_model("mobilenet_v3_large", use_pretrained=use_pretrained)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Softmax(dim=1))
    if model is None:
        raise ValueError(f"Unknown classification model_name: {model_name}")
    if verbose:
        print_model_param_count(model, model_name)
    return model

def load_model_regression(model_name, in_channels=3, use_pretrained=DEFAULT_USE_PRETRAINED, verbose=True):
    """Build a regression model by name, including ablation variants."""

    model = None
    timm_builder = TIMM_REGRESSION_BUILDERS.get(model_name)
    if timm_builder is not None:
        model = timm_builder(
            in_channels=in_channels,
            use_pretrained=use_pretrained,
        )
    if model_name == "mine":
        model = build_regression_model(in_channels=in_channels)
    if model_name in REGRESSION_ABLATION_CONFIGS:
        use_multiscale, use_attention, use_residual = REGRESSION_ABLATION_CONFIGS[model_name]
        model = build_regression_ablation_model(
            in_channels=in_channels,
            use_multiscale=use_multiscale,
            use_attention=use_attention,
            use_residual=use_residual,
        )
    if model_name == "mine_plain_cnn":
        model = build_regression_plain_cnn_model(in_channels=in_channels)
    if model_name == "resnet50":
        model = _build_torchvision_model("resnet50", use_pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs//2), nn.ReLU(), nn.Linear(num_ftrs//2, 1),  nn.Sigmoid())
    if model_name == "resnet18":
        model = _build_torchvision_model("resnet18", use_pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs//2), nn.ReLU(), nn.Linear(num_ftrs//2, 1),  nn.Sigmoid())
    if model_name == "vgg19":
        # Replace the final classifier layer by key in the nn.Module container.
        model = _build_torchvision_model("vgg19", use_pretrained=use_pretrained)
        model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())
    if model_name == "AlexNet":
        model = _build_torchvision_model("AlexNet", use_pretrained=use_pretrained)
        model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())
    if model_name == "vgg16":
        # Replace the final classifier layer by key in the nn.Module container.
        model = _build_torchvision_model("vgg19", use_pretrained=use_pretrained)
        model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())
    if model_name == "googleNet":
        # Replace the final fully connected layer with a one-output regressor.
        model = _build_torchvision_model("googleNet", use_pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    if model is None:
        raise ValueError(f"Unknown regression model_name: {model_name}")
    if verbose:
        print_model_param_count(model, model_name)
    return model


def normalize_ppm(labels):
    """Map ppm labels from [REGRESSION_PPM_MIN, REGRESSION_PPM_MAX] to [0, 1]."""

    return (labels - REGRESSION_PPM_MIN) / (REGRESSION_PPM_MAX - REGRESSION_PPM_MIN)


def denormalize_ppm(outputs):
    """Map normalized regression outputs back to ppm values."""

    return outputs * (REGRESSION_PPM_MAX - REGRESSION_PPM_MIN) + REGRESSION_PPM_MIN


def _synchronize_device(device):
    """Synchronize CUDA before and after timing when needed."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_single_sample_inference_time(model, sample_tensor, device, warmup_runs=10, timed_runs=50):
    """
    Measure average single-sample inference latency in seconds.

    The reported value is the mean time of one forward pass after a short
    warm-up phase.
    """

    model.eval()
    sample_tensor = sample_tensor.to(device)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_tensor)

        _synchronize_device(device)
        start_time = time.perf_counter()
        for _ in range(timed_runs):
            _ = model(sample_tensor)
        _synchronize_device(device)
        end_time = time.perf_counter()
    return (end_time - start_time) / timed_runs


def _train_test_classification_with_datasets(
    model_name,
    epoch_num,
    batch_size,
    lr,
    graph_type,
    train_dataset,
    test_dataset,
    in_channels=3,
    reduce_method="pca",
    label_names=None,
    class_names=None,
    random_seed=None,
    save_metric_plots=True,
    save_results_csv=True,
    save_confusion_figure=True,
    save_cluster_figure=True,
    results_dir_name="train_test_results",
    confusion_dir_name="confusion_matrix",
    cluster_dir_name="cluster",
    artifact_tag=None,
):
    """Shared classification train/test loop used by different dataset sources."""

    if class_names is None:
        class_names = getattr(train_dataset, "class_names", gas_name_array)
    if label_names is None:
        label_names = resolve_confusion_matrix_label_names(class_names)
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_name,
        in_channels=in_channels,
        num_classes=len(class_names),
    )
    model = model.to(device)

    train_generator = None
    if random_seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(random_seed)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.95)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    test_loss_meter = AverageMeter()
    test_accuracy_meter = AverageMeter()
    train_losses_total = []
    train_accuracy_total = []
    test_losses_total = []
    test_accuracy_total = []
    epoch_list = []
    final_epoch_label_true = []
    final_epoch_label_pred = []

    training_start_time = time.perf_counter()
    for epoch in range(epoch_num):
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        test_loss_meter.reset()
        test_accuracy_meter.reset()
        final_epoch_label_true = []
        final_epoch_label_pred = []

        model.train()
        for inputs, labels in train_data_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.item(), inputs.size(0))
            train_accuracy_meter.update(
                accuracy_score(labels, outputs.argmax(1).cpu().numpy()),
                inputs.size(0),
            )

        scheduler.step()
        train_losses_total.append(train_loss_meter.avg)
        train_accuracy_total.append(train_accuracy_meter.avg)

        model.eval()
        with torch.no_grad():
            for inputs, labels in test_data_loader:
                outputs = model(inputs.to(device))
                loss = loss_fn(outputs, labels.to(device))
                predictions = outputs.argmax(1).cpu().numpy()
                test_loss_meter.update(loss.item(), inputs.size(0))
                test_accuracy_meter.update(accuracy_score(labels, predictions), inputs.size(0))
                if epoch == epoch_num - 1:
                    final_epoch_label_true.append(labels.cpu().numpy())
                    final_epoch_label_pred.append(predictions)

        test_losses_total.append(test_loss_meter.avg)
        test_accuracy_total.append(test_accuracy_meter.avg)
        print(
            f"\t Epoch {epoch + 1} / {epoch_num}, train loss: {train_loss_meter.avg}, "
            f"test loss: {test_loss_meter.avg}, train accuracy: {train_accuracy_meter.avg}, "
            f"test accuracy: {test_accuracy_meter.avg}"
        )
        epoch_list.append(epoch + 1)

    _synchronize_device(device)
    training_time_seconds = time.perf_counter() - training_start_time

    inference_sample, _ = test_dataset[0]
    inference_sample = inference_sample.unsqueeze(0)
    inference_time_seconds = measure_single_sample_inference_time(model, inference_sample, device)

    print(f"[Timing] Training Time (S): {training_time_seconds:.6f}")
    print(f"[Timing] Inference Time (S): {inference_time_seconds:.6f}")

    if save_metric_plots:
        plot_train_test_figure(
            (12, 8),
            epoch_list,
            train_losses_total,
            test_losses_total,
            xlabel="epoch",
            ylabel="Loss",
            labels=["train loss", "test loss"],
            colors=["darkblue", "darkred"],
        )
        plot_train_test_figure(
            (12, 8),
            epoch_list,
            train_accuracy_total,
            test_accuracy_total,
            xlabel="epoch",
            ylabel="Accuracy",
            labels=["train accuracy", "test accuracy"],
            colors=["darkblue", "darkred"],
        )
    results_csv_path = None
    if save_results_csv:
        results_csv_path = save_results(
            model_name,
            graph_type,
            batch_size,
            lr,
            epoch_list,
            train_losses_total,
            test_losses_total,
            train_accuracy_total,
            test_accuracy_total,
            training_time_seconds=training_time_seconds,
            inference_time_seconds=inference_time_seconds,
            results_dir_name=results_dir_name,
            artifact_tag=artifact_tag,
        )
    confusion_path = None
    if save_confusion_figure:
        confusion_path = save_confusion_matrix_figure(
            label_true=final_epoch_label_true,
            label_pred=final_epoch_label_pred,
            model_name=model_name,
            graph_type=graph_type,
            epoch_num=epoch_num,
            label_names=label_names,
            output_dir_name=confusion_dir_name,
            artifact_tag=artifact_tag,
        )
    if save_cluster_figure:
        export_last_fc_clusters(
            model=model,
            model_name=model_name,
            graph_type=graph_type,
            batch_size=batch_size,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
            reduce_method=reduce_method,
            class_names=class_names,
            output_dir_name=cluster_dir_name,
            artifact_tag=artifact_tag,
        )

    return {
        "model_name": model_name,
        "graph_type": graph_type,
        "batch_size": batch_size,
        "lr": lr,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "epoch_list": epoch_list,
        "train_losses_total": train_losses_total,
        "test_losses_total": test_losses_total,
        "train_accuracy_total": train_accuracy_total,
        "test_accuracy_total": test_accuracy_total,
        "training_time_seconds": training_time_seconds,
        "inference_time_seconds": inference_time_seconds,
        "results_csv_path": results_csv_path,
        "confusion_path": confusion_path,
        "class_names": class_names,
        "label_names": label_names,
    }


def train_test_classification(model_name, epoch_num, batch_size, lr, graph_type, in_channels=3, reduce_method="pca"):
    """Train and evaluate one classification model, then save metric curves."""

    train_dataset = MyDataset(base_data_path, "train", transform, gas_name_array, gas_label, graph_type)
    test_dataset = MyDataset(base_data_path, "test", transform, gas_name_array, gas_label, graph_type)
    _train_test_classification_with_datasets(
        model_name=model_name,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        graph_type=graph_type,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        in_channels=in_channels,
        reduce_method=reduce_method,
        label_names=resolve_confusion_matrix_label_names(gas_name_array),
        class_names=gas_name_array,
    )


def train_test_classification_generated_images(
    model_name,
    epoch_num,
    batch_size,
    lr,
    graph_type,
    in_channels=3,
    reduce_method="pca",
):
    """Train and evaluate one classification model on the generated gas-response image dataset."""

    train_dataset = GeneratedGasImageDataset(
        generated_img_base_data_path,
        "train",
        transform,
    )
    test_dataset = GeneratedGasImageDataset(
        generated_img_base_data_path,
        "test",
        transform,
    )
    class_names = train_dataset.class_names
    _train_test_classification_with_datasets(
        model_name=model_name,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        graph_type=graph_type,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        in_channels=in_channels,
        reduce_method=reduce_method,
        label_names=resolve_confusion_matrix_label_names(class_names),
        class_names=class_names,
    )


def _build_repeated_experiment_axes(figure_size, xlabel="epoch", ylabel="Loss"):
    """Create axes styled to match the five-fold plotting script."""

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


def _plot_repeated_experiment_metric(dataframe, train_column, test_column, ylabel, save_path):
    """Plot light single-run curves plus one dark mean curve for one metric."""

    fig, axes, legend_font = _build_repeated_experiment_axes((12, 8), xlabel="Epoch", ylabel=ylabel)
    run_dataframe = dataframe[dataframe["record_type"] == "run"].copy()
    run_dataframe["repeat_id"] = run_dataframe["repeat_id"].astype(int)

    train_pivot = (
        run_dataframe
        .pivot_table(index="epoch", columns="repeat_id", values=train_column, aggfunc="mean")
        .sort_index()
    )
    test_pivot = (
        run_dataframe
        .pivot_table(index="epoch", columns="repeat_id", values=test_column, aggfunc="mean")
        .sort_index()
    )

    mean_dataframe = dataframe[dataframe["record_type"] == "mean"].copy().sort_values("epoch")
    epoch_values = train_pivot.index.tolist()

    light_red = "#f3a2a2"
    dark_red = "#8b0000"
    light_blue = "#a8c9ff"
    dark_blue = "#003f8c"

    for repeat_id in train_pivot.columns:
        axes.plot(
            epoch_values,
            train_pivot[repeat_id].to_numpy(),
            color=light_red,
            linewidth=2.0,
            alpha=0.9,
        )
    for repeat_id in test_pivot.columns:
        axes.plot(
            epoch_values,
            test_pivot[repeat_id].to_numpy(),
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
    plt.show()
    plt.close(fig)


def _save_repeated_generated_image_experiment_outputs(
    model_name,
    graph_type,
    batch_size,
    lr,
    repeat_times,
    run_results,
    results_dir_name="train_test_result_2",
    summary_last_epochs=20,
    file_tag="repeat",
    log_prefix="[GeneratedRepeat]",
):
    """Save one combined CSV and two summary figures for repeated experiments."""

    results_dir = os.path.join(get_script_dir(), results_dir_name)
    os.makedirs(results_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_stem = f"{model_name}_{graph_type}_{file_tag}_{repeat_times}_{time_str}"
    csv_path = os.path.join(results_dir, file_stem + ".csv")
    loss_path = os.path.join(results_dir, file_stem + "_loss.png")
    accuracy_path = os.path.join(results_dir, file_stem + "_accuracy.png")

    rows = []
    for repeat_index, run_result in enumerate(run_results, start=1):
        for epoch_offset, epoch in enumerate(run_result["epoch_list"]):
            rows.append({
                "record_type": "run",
                "repeat_id": repeat_index,
                "epoch": epoch,
                "model_name": model_name,
                "graph_type": graph_type,
                "batch_size": batch_size,
                "lr": lr,
                "train_size": run_result["train_size"],
                "test_size": run_result["test_size"],
                "train_loss": run_result["train_losses_total"][epoch_offset],
                "test_loss": run_result["test_losses_total"][epoch_offset],
                "train_accuracy": run_result["train_accuracy_total"][epoch_offset],
                "test_accuracy": run_result["test_accuracy_total"][epoch_offset],
                "training_time_seconds": run_result["training_time_seconds"],
                "inference_time_seconds": run_result["inference_time_seconds"],
                "summary_name": "",
                "summary_last_epochs": np.nan,
                "train_loss_last_mean": np.nan,
                "train_loss_last_std": np.nan,
                "test_loss_last_mean": np.nan,
                "test_loss_last_std": np.nan,
                "train_accuracy_last_mean": np.nan,
                "train_accuracy_last_std": np.nan,
                "test_accuracy_last_mean": np.nan,
                "test_accuracy_last_std": np.nan,
                "training_time_mean": np.nan,
                "training_time_std": np.nan,
                "inference_time_mean": np.nan,
                "inference_time_std": np.nan,
            })

    dataframe = pd.DataFrame(rows)
    mean_dataframe = (
        dataframe[dataframe["record_type"] == "run"]
        .groupby("epoch", as_index=False)[["train_loss", "test_loss", "train_accuracy", "test_accuracy"]]
        .mean()
    )
    mean_dataframe["record_type"] = "mean"
    mean_dataframe["repeat_id"] = "mean"
    mean_dataframe["model_name"] = model_name
    mean_dataframe["graph_type"] = graph_type
    mean_dataframe["batch_size"] = batch_size
    mean_dataframe["lr"] = lr
    mean_dataframe["train_size"] = run_results[0]["train_size"]
    mean_dataframe["test_size"] = run_results[0]["test_size"]
    mean_dataframe["training_time_seconds"] = np.mean([item["training_time_seconds"] for item in run_results])
    mean_dataframe["inference_time_seconds"] = np.mean([item["inference_time_seconds"] for item in run_results])
    mean_dataframe["summary_name"] = ""
    mean_dataframe["summary_last_epochs"] = np.nan
    mean_dataframe["train_loss_last_mean"] = np.nan
    mean_dataframe["train_loss_last_std"] = np.nan
    mean_dataframe["test_loss_last_mean"] = np.nan
    mean_dataframe["test_loss_last_std"] = np.nan
    mean_dataframe["train_accuracy_last_mean"] = np.nan
    mean_dataframe["train_accuracy_last_std"] = np.nan
    mean_dataframe["test_accuracy_last_mean"] = np.nan
    mean_dataframe["test_accuracy_last_std"] = np.nan
    mean_dataframe["training_time_mean"] = np.nan
    mean_dataframe["training_time_std"] = np.nan
    mean_dataframe["inference_time_mean"] = np.nan
    mean_dataframe["inference_time_std"] = np.nan

    tail_run_dataframe = (
        dataframe[dataframe["record_type"] == "run"]
        .groupby("repeat_id", group_keys=False)
        .tail(summary_last_epochs)
        .copy()
    )
    summary_row = {
        "record_type": "summary",
        "repeat_id": "summary",
        "epoch": np.nan,
        "model_name": model_name,
        "graph_type": graph_type,
        "batch_size": batch_size,
        "lr": lr,
        "train_size": run_results[0]["train_size"],
        "test_size": run_results[0]["test_size"],
        "train_loss": np.nan,
        "test_loss": np.nan,
        "train_accuracy": np.nan,
        "test_accuracy": np.nan,
        "training_time_seconds": np.nan,
        "inference_time_seconds": np.nan,
        "summary_name": "last_{}_epochs_across_{}_runs".format(summary_last_epochs, repeat_times),
        "summary_last_epochs": int(summary_last_epochs),
        "train_loss_last_mean": float(tail_run_dataframe["train_loss"].mean()),
        "train_loss_last_std": float(tail_run_dataframe["train_loss"].std(ddof=0)),
        "test_loss_last_mean": float(tail_run_dataframe["test_loss"].mean()),
        "test_loss_last_std": float(tail_run_dataframe["test_loss"].std(ddof=0)),
        "train_accuracy_last_mean": float(tail_run_dataframe["train_accuracy"].mean()),
        "train_accuracy_last_std": float(tail_run_dataframe["train_accuracy"].std(ddof=0)),
        "test_accuracy_last_mean": float(tail_run_dataframe["test_accuracy"].mean()),
        "test_accuracy_last_std": float(tail_run_dataframe["test_accuracy"].std(ddof=0)),
        "training_time_mean": float(np.mean([item["training_time_seconds"] for item in run_results])),
        "training_time_std": float(np.std([item["training_time_seconds"] for item in run_results], ddof=0)),
        "inference_time_mean": float(np.mean([item["inference_time_seconds"] for item in run_results])),
        "inference_time_std": float(np.std([item["inference_time_seconds"] for item in run_results], ddof=0)),
    }

    final_dataframe = pd.concat(
        [dataframe, mean_dataframe, pd.DataFrame([summary_row])],
        ignore_index=True,
        sort=False,
    )
    final_dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")

    _plot_repeated_experiment_metric(
        dataframe=final_dataframe,
        train_column="train_loss",
        test_column="test_loss",
        ylabel="Loss",
        save_path=loss_path,
    )
    _plot_repeated_experiment_metric(
        dataframe=final_dataframe,
        train_column="train_accuracy",
        test_column="test_accuracy",
        ylabel="Accuracy",
        save_path=accuracy_path,
    )

    print(f"{log_prefix} Saved combined CSV to: {csv_path}")
    print(f"{log_prefix} Saved loss figure to: {loss_path}")
    print(f"{log_prefix} Saved accuracy figure to: {accuracy_path}")
    return csv_path, loss_path, accuracy_path


def repeat_train_test_classification_generated_images(
    model_name,
    epoch_num,
    batch_size,
    lr,
    graph_type,
    in_channels=3,
    reduce_method="pca",
    repeat_times=5,
):
    """Repeat generated-image classification experiments and save one combined CSV plus summary figures."""

    if repeat_times <= 0:
        raise ValueError("repeat_times must be greater than 0.")

    train_dataset = GeneratedGasImageDataset(
        generated_img_base_data_path,
        "train",
        transform,
    )
    test_dataset = GeneratedGasImageDataset(
        generated_img_base_data_path,
        "test",
        transform,
    )
    class_names = train_dataset.class_names
    run_results = []
    for repeat_index in range(1, repeat_times + 1):
        run_seed = 42 + repeat_index - 1
        print(f"[GeneratedRepeat] run {repeat_index}/{repeat_times}, seed={run_seed}")
        run_result = _train_test_classification_with_datasets(
            model_name=model_name,
            epoch_num=epoch_num,
            batch_size=batch_size,
            lr=lr,
            graph_type=graph_type,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            in_channels=in_channels,
            reduce_method=reduce_method,
            label_names=resolve_confusion_matrix_label_names(class_names),
            class_names=class_names,
            random_seed=run_seed,
            save_metric_plots=False,
            save_results_csv=False,
            save_confusion_figure=True,
            save_cluster_figure=True,
            results_dir_name="train_test_result_2",
            confusion_dir_name="confusion_matrix_2",
            cluster_dir_name="cluster_2",
            artifact_tag=f"run{repeat_index}",
        )
        run_results.append(run_result)

    return _save_repeated_generated_image_experiment_outputs(
        model_name=model_name,
        graph_type=graph_type,
        batch_size=batch_size,
        lr=lr,
        repeat_times=repeat_times,
        run_results=run_results,
        results_dir_name="train_test_result_2",
    )


def train_test_regression(model_name, epoch_num, batch_size, lr, graph_type, in_channels=3):
    """Train and evaluate one ppm regression model, then save metric curves."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_regression(model_name, in_channels=in_channels)
    model = model.to(device)
    train_Dataset = MyDataset_regression(base_data_path, "train", transform, gas_name_array, gas_label, graph_type)
    test_Dataset = MyDataset_regression(base_data_path, "test", transform, gas_name_array, gas_label, graph_type)

    train_data_loader = DataLoader(train_Dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(test_Dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr * 0.05)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.05)
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    test_loss_meter = AverageMeter()
    test_accuracy_meter = AverageMeter()
    train_losses_total = []
    train_accuracy_total = []
    test_losses_total = []
    test_accuracy_total = []
    epoch_list = []
    for epoch in range(epoch_num):
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        test_loss_meter.reset()
        test_accuracy_meter.reset()
        train_outputs_total = []
        train_labels_total = []
        test_outputs_total = []
        test_labels_total = []
        model.train()
        for inputs, labels in train_data_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            labels = labels.float().to(device)
            outputs = outputs.view(-1)
            labels_norm = normalize_ppm(labels)
            loss = loss_fn(outputs, labels_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            outputs_ppm = denormalize_ppm(outputs)
            train_loss_meter.update(torch.nn.functional.l1_loss(outputs_ppm, labels).item(), inputs.size(0))
            train_outputs_total.append(outputs_ppm.detach().cpu())
            train_labels_total.append(labels.detach().cpu())
        scheduler.step()
        train_r2 = r2_score(torch.cat(train_labels_total).numpy(), torch.cat(train_outputs_total).numpy())
        train_losses_total.append(train_loss_meter.avg)
        train_accuracy_total.append(train_r2)
        # test
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_data_loader:
                outputs = model(inputs.to(device))
                labels = labels.float().to(device)
                outputs = outputs.view(-1)
                outputs_ppm = denormalize_ppm(outputs)
                loss = torch.nn.functional.l1_loss(outputs_ppm, labels)
                test_loss_meter.update(loss.item(), inputs.size(0))
                test_outputs_total.append(outputs_ppm.cpu())
                test_labels_total.append(labels.cpu())
        test_r2 = r2_score(torch.cat(test_labels_total).numpy(), torch.cat(test_outputs_total).numpy())
        test_losses_total.append(test_loss_meter.avg)
        test_accuracy_total.append(test_r2)

        print(
            f"\t Epoch {epoch + 1} / {epoch_num}, train loss: {train_loss_meter.avg}, test loss: {test_loss_meter.avg}, train R2: {train_r2}, test R2: {test_r2}")
        epoch_list.append(epoch + 1)
        if epoch == epoch_num - 1:
            plot_train_test_figure((12, 8), epoch_list, train_losses_total, test_losses_total, xlabel="epoch",
                                   ylabel="Loss", labels=["train loss", "test loss"], colors=["darkblue", "darkred"])
            plot_train_test_figure((12, 8), epoch_list, train_accuracy_total, test_accuracy_total, xlabel="epoch",
                                   ylabel="R2", labels=["train R2", "test R2"],
                                   colors=["darkblue", "darkred"])
            save_results_regression(model_name, graph_type, batch_size, lr, epoch_list, train_losses_total, test_losses_total, train_accuracy_total, test_accuracy_total)


def save_results(
    model_name,
    graph_type,
    batch_size,
    lr,
    epoch_list,
    train_losses_total,
    test_losses_total,
    train_accuracy_total,
    test_accuracy_total,
    training_time_seconds=None,
    inference_time_seconds=None,
    results_dir_name="train_test_results",
    artifact_tag=None,
):
    """Save classification training and test metrics to a timestamped CSV file."""

    dataframe = pd.DataFrame({
        'epoch': epoch_list,
        'train loss': train_losses_total,
        'test loss': test_losses_total,
        'train accuracy': train_accuracy_total,
        'test accuracy': test_accuracy_total,
        'batch size': batch_size,
        'lr': lr,
        'Training Time (S)': training_time_seconds,
        'Inference Time (S)': inference_time_seconds,
    })
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    results_dir = os.path.join(get_script_dir(), results_dir_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    file_stem = model_name + "_" + graph_type
    if artifact_tag:
        file_stem += "_" + artifact_tag
    save_path = os.path.join(results_dir, file_stem + "_" + time_str + ".csv")
    print(save_path)
    dataframe.to_csv(save_path, index=False)
    return save_path


def save_results_regression(model_name, graph_type, batch_size, lr, epoch_list, train_losses_total, test_losses_total, train_r2_total, test_r2_total):
    """Save regression training and test metrics to a timestamped CSV file."""

    dataframe = pd.DataFrame({'epoch': epoch_list, 'train loss': train_losses_total, 'test loss': test_losses_total, 'train R2': train_r2_total, 'test R2': test_r2_total, 'batch size': batch_size, 'lr':lr})
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    results_dir = './train_test_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_path = results_dir+'/'+model_name+"_"+graph_type+"_regression_"+time_str+".csv"
    print(save_path)
    dataframe.to_csv(save_path, index=False)


def save_confusion_matrix_figure(
    label_true,
    label_pred,
    model_name,
    graph_type,
    epoch_num,
    label_names=None,
    output_dir_name="confusion_matrix",
    artifact_tag=None,
):
    """Save and display the final-epoch confusion matrix with the reference style."""

    if label_names is None:
        label_names = resolve_confusion_matrix_label_names(gas_name_array)
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(get_script_dir(), output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    file_stem = f"{model_name}_{graph_type}_epoch{epoch_num}_confusion_matrix"
    if artifact_tag:
        file_stem += f"_{artifact_tag}"
    save_path = os.path.join(
        output_dir,
        f"{file_stem}_{time_str}.png",
    )
    draw_confusion_matrix(
        label_true=label_true,
        label_pred=label_pred,
        label_name=label_names,
        title="Confusion Matrix",
        pdf_save_path=save_path,
        dpi=300,
    )
    print(f"[ConfusionMatrix] Saved figure to: {save_path}")
    return save_path


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """Plot a normalized confusion matrix and optionally save it to disk.

    Args:
        label_true: Iterable of true-label arrays collected by batch.
        label_pred: Iterable of predicted-label arrays collected by batch.
        label_name: Display names for each class.
        title: Figure title kept for compatibility.
        pdf_save_path: Optional output path, such as .png or .pdf.
        dpi: Save resolution used when ``pdf_save_path`` is provided.
    """
    _label_true=[]
    _label_pred=[]
    for m in range(0,len(label_true)):
        size=label_true[m].shape[0]
        for n in range(0,size):
            _label_true.append(label_true[m][n])
    for m in range(0, len(label_pred)):
        size = label_pred[m].shape[0]
        for n in range(0, size):
            _label_pred.append(label_pred[m][n])
    # print(f'label true: {_label_true}')
    # print(f'label pred: {_label_pred}')
    # label_true=(np.array(label_true).flatten()).tolist()
    # label_pred =(np.array(label_pred).flatten()).tolist()

    # label_true = list((np.array(label_true).flatten()))
    # label_pred = list((np.array(label_pred).flatten()))


    cm = confusion_matrix(
        y_true=_label_true,
        y_pred=_label_pred,
        labels=list(range(len(label_name))),
        normalize='true',
    )
    axis_title_font = {'family': 'Arial', 'weight': 'bold'}
    tick_font = FontProperties(family='Arial', weight='bold')

    is_large_twelve_class_plot = len(label_name) == 12
    if is_large_twelve_class_plot:
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_axes([0.03, 0.10, 0.78, 0.86])
        image = ax.imshow(cm, cmap='Oranges')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Predict label", fontsize=15, labelpad=8, **axis_title_font)
        ax.set_ylabel("Truth label", fontsize=15, labelpad=4, **axis_title_font)
        ax.set_yticks(range(label_name.__len__()))
        ax.set_yticklabels(label_name, fontproperties=tick_font, fontsize=10)
        ax.set_xticks(range(label_name.__len__()))
        ax.set_xticklabels(label_name, fontproperties=tick_font, fontsize=10, rotation=0, ha='center')
        ax.tick_params(axis='x', pad=6, length=0)
        ax.tick_params(axis='y', pad=1, length=0)
    else:
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111)
        image = ax.imshow(cm, cmap='Oranges')
        ax.set_xlabel("Predict label", fontsize=22, labelpad=10, **axis_title_font)
        ax.set_ylabel("Truth label", fontsize=22, labelpad=15, **axis_title_font)
        ax.set_yticks(range(label_name.__len__()))
        ax.set_yticklabels(label_name, fontproperties=tick_font, fontsize=16)
        ax.set_xticks(range(label_name.__len__()))
        ax.set_xticklabels(label_name, fontproperties=tick_font, fontsize=16)
        plt.subplots_adjust(left=0.0002, right=1, top=0.95, bottom=0.2)

    #plt.tight_layout()

    if is_large_twelve_class_plot:
        cax = fig.add_axes([0.82, 0.10, 0.025, 0.86])
        cbar = fig.colorbar(image, cax=cax)
    else:
        cbar = fig.colorbar(image, ax=ax, fraction=0.15, pad=0.04)
    cbar.mappable.set_clim(0.0, 1.0)
    cbar.ax.tick_params(labelsize=16 if not is_large_twelve_class_plot else 12)
    #cbar.set_ticks([0.0,0.2,0.4,0.6,0.8,1.0])
    #cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    y_label = cbar.ax.get_yticklabels()
    for y_label_temp in y_label:
        y_label_temp.set_fontproperties(tick_font)
    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
             # Use white text on dark cells and black text on light cells.
            value = float(format('%.2f' % cm[j, i]))
            color = (1, 1, 1) if value > 0.5 else (0, 0, 0)
            ax.text(
                i,
                j,
                value,
                verticalalignment='center',
                horizontalalignment='center',
                color=color,
                weight='bold',
                size=9 if is_large_twelve_class_plot else 12,
            )


    # plt.show()
    if not pdf_save_path is None:
        if is_large_twelve_class_plot:
            fig.savefig(pdf_save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        else:
            fig.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

    if not is_large_twelve_class_plot:
        plt.tight_layout(pad=0.1)
    plt.show()
    plt.close(fig)


class UnifiedClassificationDataset(torch.utils.data.Dataset):
    """Merge the original train/test classification folders into one dataset."""

    def __init__(self, base_data_path, transform=None, gas_name_array=None, gas_label=None, img_type="img"):
        """Collect all classification samples from both train and test folders."""

        img_list = []
        label_list = []
        for gas_name in gas_name_array:
            for set_type in ["train", "test"]:
                set_data_path = base_data_path + "\\" + gas_name + "\\" + set_type + "_set_" + img_type
                for path, dir_lst, file_lst in os.walk(set_data_path):
                    dir_lst.sort()
                    for file_name in sorted(file_lst):
                        ppm = file_name.split('_')[1]
                        ppm_dir = ppm[:-3] + "_ppm"
                        img_data_path = set_data_path + "\\" + ppm_dir + "\\" + file_name
                        img_list.append(img_data_path)
                        label_list.append(gas_label[gas_name])
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        print(f"[KFoldDataset] merged sample count: {len(self.img_list)}")

    def __len__(self):
        """Return the total number of merged samples."""

        return len(self.img_list)

    def __getitem__(self, index):
        """Load one merged classification sample."""

        from PIL import Image as pil_image

        img = pil_image.open(self.img_list[index])
        img = self.transform(img)
        label = self.label_list[index]
        return img, label


class IndexedClassificationDataset(torch.utils.data.Dataset):
    """Subset wrapper that preserves img paths and labels for artifact export."""

    def __init__(self, base_dataset, indices, class_names=None):
        """Wrap one dataset with an explicit index list."""

        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.img_list = [base_dataset.img_list[index] for index in self.indices]
        self.label_list = [base_dataset.label_list[index] for index in self.indices]
        self.class_names = list(class_names) if class_names is not None else getattr(base_dataset, "class_names", None)

    def __len__(self):
        """Return subset size."""

        return len(self.indices)

    def __getitem__(self, index):
        """Load one sample from the wrapped base dataset."""

        return self.base_dataset[self.indices[index]]


def save_five_fold_classification_results(
    model_name,
    graph_type,
    batch_size,
    lr,
    epoch_num,
    epoch_rows,
    fold_rows,
    file_tag="five_fold",
    results_dir_name="five_fold_cv_results",
    log_prefix="[FiveFoldCV]",
):
    """Save detailed and summary CSV files for repeated classification experiments."""

    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    results_dir = os.path.join(get_script_dir(), results_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    epoch_dataframe = pd.DataFrame(epoch_rows)
    epoch_save_path = os.path.join(
        results_dir,
        f"{model_name}_{graph_type}_{file_tag}_detail_{time_str}.csv",
    )
    epoch_dataframe.to_csv(epoch_save_path, index=False, encoding="utf-8-sig")

    fold_dataframe = pd.DataFrame(fold_rows)
    summary_row = {
        "fold": "mean",
        "model_name": model_name,
        "graph_type": graph_type,
        "epoch_num": epoch_num,
        "batch_size": batch_size,
        "lr": lr,
        "train_size": float(fold_dataframe["train_size"].mean()),
        "val_size": float(fold_dataframe["val_size"].mean()),
        "best_epoch": float(fold_dataframe["best_epoch"].mean()),
        "best_train_loss": float(fold_dataframe["best_train_loss"].mean()),
        "best_val_loss": float(fold_dataframe["best_val_loss"].mean()),
        "best_train_accuracy": float(fold_dataframe["best_train_accuracy"].mean()),
        "best_val_accuracy": float(fold_dataframe["best_val_accuracy"].mean()),
        "final_train_loss": float(fold_dataframe["final_train_loss"].mean()),
        "final_val_loss": float(fold_dataframe["final_val_loss"].mean()),
        "final_train_accuracy": float(fold_dataframe["final_train_accuracy"].mean()),
        "final_val_accuracy": float(fold_dataframe["final_val_accuracy"].mean()),
    }
    std_row = {
        "fold": "std",
        "model_name": model_name,
        "graph_type": graph_type,
        "epoch_num": epoch_num,
        "batch_size": batch_size,
        "lr": lr,
        "train_size": float(fold_dataframe["train_size"].std(ddof=0)),
        "val_size": float(fold_dataframe["val_size"].std(ddof=0)),
        "best_epoch": float(fold_dataframe["best_epoch"].std(ddof=0)),
        "best_train_loss": float(fold_dataframe["best_train_loss"].std(ddof=0)),
        "best_val_loss": float(fold_dataframe["best_val_loss"].std(ddof=0)),
        "best_train_accuracy": float(fold_dataframe["best_train_accuracy"].std(ddof=0)),
        "best_val_accuracy": float(fold_dataframe["best_val_accuracy"].std(ddof=0)),
        "final_train_loss": float(fold_dataframe["final_train_loss"].std(ddof=0)),
        "final_val_loss": float(fold_dataframe["final_val_loss"].std(ddof=0)),
        "final_train_accuracy": float(fold_dataframe["final_train_accuracy"].std(ddof=0)),
        "final_val_accuracy": float(fold_dataframe["final_val_accuracy"].std(ddof=0)),
    }
    summary_dataframe = pd.concat(
        [fold_dataframe, pd.DataFrame([summary_row, std_row])],
        ignore_index=True,
    )
    summary_save_path = os.path.join(
        results_dir,
        f"{model_name}_{graph_type}_{file_tag}_summary_{time_str}.csv",
    )
    summary_dataframe.to_csv(summary_save_path, index=False, encoding="utf-8-sig")

    print(f"{log_prefix} Saved detailed CSV to: {epoch_save_path}")
    print(f"{log_prefix} Saved summary CSV to: {summary_save_path}")
    return epoch_save_path, summary_save_path


def five_fold_cross_validation_classification(
    model_name,
    epoch_num,
    batch_size,
    lr,
    graph_type,
    in_channels=3,
    n_splits=5,
    random_state=42,
):
    """Run classification five-fold cross validation on merged train/test data."""

    from sklearn.model_selection import StratifiedKFold

    if n_splits != 5:
        raise ValueError("This function is fixed to five-fold cross validation, so n_splits must be 5.")

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = UnifiedClassificationDataset(
        base_data_path=base_data_path,
        transform=transform,
        gas_name_array=gas_name_array,
        gas_label=gas_label,
        img_type=graph_type,
    )
    labels_array = np.array(full_dataset.label_list)
    sample_index_array = np.arange(len(full_dataset))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    epoch_rows = []
    fold_rows = []
    for fold_index, (train_index, val_index) in enumerate(
        splitter.split(sample_index_array, labels_array),
        start=1,
    ):
        print(f"[FiveFoldCV] fold {fold_index}/{n_splits}")
        model = load_model(model_name, in_channels=in_channels)
        model = model.to(device)

        train_subset = torch.utils.data.Subset(full_dataset, train_index.tolist())
        val_subset = torch.utils.data.Subset(full_dataset, val_index.tolist())
        train_data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_data_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.1,
        )
        scheduler = StepLR(optimizer, step_size=20, gamma=0.95)
        loss_fn = torch.nn.CrossEntropyLoss()

        fold_epoch_metrics = []
        for epoch in range(epoch_num):
            train_loss_meter = AverageMeter()
            train_accuracy_meter = AverageMeter()
            val_loss_meter = AverageMeter()
            val_accuracy_meter = AverageMeter()

            model.train()
            for inputs, labels in train_data_loader:
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                labels_device = labels.to(device)
                loss = loss_fn(outputs, labels_device)
                loss.backward()
                optimizer.step()

                predictions = outputs.argmax(1).detach().cpu().numpy()
                train_loss_meter.update(loss.item(), inputs.size(0))
                train_accuracy_meter.update(
                    accuracy_score(labels.cpu().numpy(), predictions),
                    inputs.size(0),
                )

            scheduler.step()

            model.eval()
            with torch.no_grad():
                for inputs, labels in val_data_loader:
                    outputs = model(inputs.to(device))
                    labels_device = labels.to(device)
                    loss = loss_fn(outputs, labels_device)
                    predictions = outputs.argmax(1).detach().cpu().numpy()
                    val_loss_meter.update(loss.item(), inputs.size(0))
                    val_accuracy_meter.update(
                        accuracy_score(labels.cpu().numpy(), predictions),
                        inputs.size(0),
                    )

            current_epoch_metrics = {
                "fold": fold_index,
                "epoch": epoch + 1,
                "model_name": model_name,
                "graph_type": graph_type,
                "batch_size": batch_size,
                "lr": lr,
                "train_size": len(train_index),
                "val_size": len(val_index),
                "train_loss": train_loss_meter.avg,
                "val_loss": val_loss_meter.avg,
                "train_accuracy": train_accuracy_meter.avg,
                "val_accuracy": val_accuracy_meter.avg,
            }
            fold_epoch_metrics.append(current_epoch_metrics)
            epoch_rows.append(current_epoch_metrics)

            print(
                f"\t Fold {fold_index}, Epoch {epoch + 1} / {epoch_num}, "
                f"train loss: {train_loss_meter.avg}, val loss: {val_loss_meter.avg}, "
                f"train accuracy: {train_accuracy_meter.avg}, val accuracy: {val_accuracy_meter.avg}"
            )

        best_epoch_metrics = max(
            fold_epoch_metrics,
            key=lambda item: (item["val_accuracy"], -item["val_loss"]),
        )
        final_epoch_metrics = fold_epoch_metrics[-1]
        fold_rows.append({
            "fold": fold_index,
            "model_name": model_name,
            "graph_type": graph_type,
            "epoch_num": epoch_num,
            "batch_size": batch_size,
            "lr": lr,
            "train_size": len(train_index),
            "val_size": len(val_index),
            "best_epoch": best_epoch_metrics["epoch"],
            "best_train_loss": best_epoch_metrics["train_loss"],
            "best_val_loss": best_epoch_metrics["val_loss"],
            "best_train_accuracy": best_epoch_metrics["train_accuracy"],
            "best_val_accuracy": best_epoch_metrics["val_accuracy"],
            "final_train_loss": final_epoch_metrics["train_loss"],
            "final_val_loss": final_epoch_metrics["val_loss"],
            "final_train_accuracy": final_epoch_metrics["train_accuracy"],
            "final_val_accuracy": final_epoch_metrics["val_accuracy"],
        })

    return save_five_fold_classification_results(
        model_name=model_name,
        graph_type=graph_type,
        batch_size=batch_size,
        lr=lr,
        epoch_num=epoch_num,
        epoch_rows=epoch_rows,
        fold_rows=fold_rows,
    )


def five_times_validation_classification(
    model_name,
    epoch_num,
    batch_size,
    lr,
    graph_type,
    in_channels=3,
    repeat_times=5,
    random_state=42,
):
    """Run five repeated classification experiments on the fixed train/test split."""

    return repeat_train_test_classification(
        model_name=model_name,
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        graph_type=graph_type,
        in_channels=in_channels,
        reduce_method="pca",
        repeat_times=repeat_times,
        random_state=random_state,
    )


def repeat_train_test_classification(
    model_name,
    epoch_num,
    batch_size,
    lr,
    graph_type,
    in_channels=3,
    reduce_method="pca",
    repeat_times=5,
    random_state=42,
    is_large_test=False,
    use_cross_validation=False,
):
    """Run fixed-split repeats or five-fold CV on dataset one with the same artifact output style."""

    if repeat_times <= 0:
        raise ValueError("repeat_times must be greater than 0.")

    class_names, class_label_map, display_names = get_dataset_one_classification_config(is_large_test=is_large_test)
    run_results = []
    if use_cross_validation:
        from sklearn.model_selection import StratifiedKFold

        full_dataset = UnifiedClassificationDataset(
            base_data_path=base_data_path,
            transform=transform,
            gas_name_array=class_names,
            gas_label=class_label_map,
            img_type=graph_type,
        )
        labels_array = np.array(full_dataset.label_list)
        sample_index_array = np.arange(len(full_dataset))
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        effective_repeat_times = 5

        for fold_index, (train_index, val_index) in enumerate(
            splitter.split(sample_index_array, labels_array),
            start=1,
        ):
            run_seed = random_state + fold_index - 1
            print(f"[RepeatClassification] fold {fold_index}/{effective_repeat_times}, seed={run_seed}")
            train_dataset = IndexedClassificationDataset(full_dataset, train_index.tolist(), class_names=class_names)
            test_dataset = IndexedClassificationDataset(full_dataset, val_index.tolist(), class_names=class_names)
            run_result = _train_test_classification_with_datasets(
                model_name=model_name,
                epoch_num=epoch_num,
                batch_size=batch_size,
                lr=lr,
                graph_type=graph_type,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                in_channels=in_channels,
                reduce_method=reduce_method,
                label_names=display_names,
                class_names=class_names,
                random_seed=run_seed,
                save_metric_plots=False,
                save_results_csv=False,
                save_confusion_figure=True,
                save_cluster_figure=True,
                results_dir_name="train_test_results",
                confusion_dir_name="confusion_matrix",
                cluster_dir_name="cluster",
                artifact_tag=f"fold{fold_index}",
            )
            run_results.append(run_result)
        file_tag = "five_fold_cv"
    else:
        train_dataset = MyDataset(base_data_path, "train", transform, class_names, class_label_map, graph_type)
        test_dataset = MyDataset(base_data_path, "test", transform, class_names, class_label_map, graph_type)
        effective_repeat_times = repeat_times
        for repeat_index in range(1, repeat_times + 1):
            run_seed = random_state + repeat_index - 1
            print(f"[RepeatClassification] run {repeat_index}/{repeat_times}, seed={run_seed}")
            run_result = _train_test_classification_with_datasets(
                model_name=model_name,
                epoch_num=epoch_num,
                batch_size=batch_size,
                lr=lr,
                graph_type=graph_type,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                in_channels=in_channels,
                reduce_method=reduce_method,
                label_names=display_names,
                class_names=class_names,
                random_seed=run_seed,
                save_metric_plots=False,
                save_results_csv=False,
                save_confusion_figure=True,
                save_cluster_figure=True,
                results_dir_name="train_test_results",
                confusion_dir_name="confusion_matrix",
                cluster_dir_name="cluster",
                artifact_tag=f"run{repeat_index}",
            )
            run_results.append(run_result)
        file_tag = "repeat"

    return _save_repeated_generated_image_experiment_outputs(
        model_name=model_name,
        graph_type=graph_type,
        batch_size=batch_size,
        lr=lr,
        repeat_times=effective_repeat_times,
        run_results=run_results,
        results_dir_name="train_test_results",
        file_tag=file_tag,
        log_prefix="[RepeatClassification]",
    )



if __name__=="__main__":
    #print_selected_model_param_counts(model_list[4], in_channels=3, use_pretrained=False)

    repeat_train_test_classification(
        model_name=model_list[4],
        epoch_num=120,
        batch_size=48,
        lr=0.00006,
        graph_type=img[0],
        in_channels=3,
        reduce_method="tsne",
        repeat_times=5,
        is_large_test=True,
        use_cross_validation=True
    )

    #GASF
    #train_test_classification(model_list[4], 120, 48, 0.00005, img[1], in_channels=3, reduce_method="tSNE")
    # five_fold_cross_validation_classification(
    #     model_name=model_list[-3],
    #     epoch_num=120,
    #     batch_size=48,
    #     lr=0.00001,
    #     graph_type=img[0],
    #     in_channels=3,
    # )
    # train_test_classification_generated_images(
    #     model_name=model_list[14],
    #     epoch_num=150,
    #     batch_size=48,
    #     lr=0.00001,
    #     graph_type="gasf_img_v2",
    #     in_channels=3,
    #     reduce_method="tsne",)

    # repeat_train_test_classification_generated_images(
    #     model_name=model_list[-1],
    #     epoch_num=180,
    #     batch_size=48,
    #     lr=0.00008,
    #     graph_type="generated_gasf",
    #     in_channels=3,
    #     reduce_method="tsne",
    #     repeat_times=5,
    # )
    #第二个数据集，-1: 8e-5, -2:4e-5, -3:4e-5
    #[mine:0.00002, plain: 0
    #MTF
    #train_test_classification(model_list[4], 120, 48, 0.00004, img[2])

    #train_test_classification(model_list[8], 120, 48, 0.00004, img[2])

    #train_test_regression(model_list[4], 180, 32, 0.0001, img[0])

    # five_times_validation_classification(
    #     model_name=model_list[7],
    #     epoch_num=120,
    #     batch_size=48,
    #     lr=0.00005,
    #     graph_type=img[0],
    #     in_channels=3,
    # )

    #analyze_model_list_complexity(model_list)

    #for img_type in range(0, 3, 1):
        #print(img_type)
        # train_test(model_list[4], 180, 48, 0.00001, img[img_type])
        # train_test(model_list[4], 180, 48, 0.00001, white_noise_img[img_type])
        #train_test(model_list[0], 100, 48, 0.000005, img[img_type])
        #train_test(model_list[0], 100, 48, 0.000005, white_noise_img[img_type])
    #train_test_for_fig_regression(model_list[0], 100, 48, 0.000006, comparison_list[0], comparison_list[-2])

    # for model_name in model_list:
    #     load_model_regression(model_name)
