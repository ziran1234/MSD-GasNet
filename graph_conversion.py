import os

import cv2
import numpy as np
import pandas as pd
from pyts.image import MarkovTransitionField, RecurrencePlot

gas_name_array = ["1-butanol", "acetone", "benzaldehyde", "butyl acetate", "dimethylbenzene", "ethanol", "ethyl acetate", "isopropanol", "methanol", "methylbenzene", "n-heptane", "n-propanol"]
# gas_name_array = ["methanol", "methylbenzene", "n-heptane", "n-propanol"]
#gas_name_array = ["1-butanol", "acetone", "benzaldehyde", "butyl acetate", "dimethylbenzene"]
gas_dict = {
    "1-butanol": 0,
    "acetone": 0,
    "benzaldehyde": 600,
    "butyl acetate": 1100,
    "dimethylbenzene": 0,
    "ethanol": 0,
    "ethyl acetate": 0,
    "isopropanol": 560,
    "methanol": 0,
    "methylbenzene": 0,
    "n-heptane": 0,
    "n-propanol": 0,
}
data_type = ["average", "white_noise", "peak_noise", "drift", "noise"]
base_data_path = r"D:\E-nose_DataSet\Gas_Recognition_DataSet\gas_response_data"
cv2_colormap = 6


def _import_pywt():
    """Import pywt lazily so only CWT generation requires the dependency."""

    try:
        import pywt
    except ImportError as exc:
        raise ImportError("CWT conversion requires the 'PyWavelets' package.") from exc
    return pywt


def _import_scipy_signal():
    """Import scipy.signal lazily so only STFT generation requires the dependency."""

    try:
        from scipy import signal
    except ImportError as exc:
        raise ImportError("STFT conversion requires the 'scipy' package.") from exc
    return signal


def get_output_graph_type(graph_type, save_gray=False):
    """Append ``gray`` to output folder names when grayscale images are saved."""

    if not save_gray:
        return graph_type
    if "gray" in graph_type:
        return graph_type
    if graph_type.endswith("_img"):
        return graph_type[:-4] + "_gray_img"
    return graph_type + "_gray"


def build_output_image(normalized_matrix, img_size=None, save_gray=False):
    """Return either a grayscale matrix or a color-mapped image for saving."""

    final_img = normalized_matrix if save_gray else cv2.applyColorMap(normalized_matrix, cv2_colormap)
    if img_size is not None:
        final_img = cv2.resize(final_img, (img_size, img_size))
    return final_img


def _normalize_matrix_to_uint8(matrix):
    """Scale one matrix into the uint8 range used by the saved image files."""

    matrix = np.asarray(matrix, dtype=np.float32)
    min_val = float(np.min(matrix))
    max_val = float(np.max(matrix))
    if max_val == min_val:
        return np.zeros_like(matrix, dtype=np.uint8)
    normalized_matrix = (matrix - min_val) / (max_val - min_val) * 255.0
    return normalized_matrix.astype(np.uint8)


def _iter_source_csv_files(base_data_path):
    """Yield gas name, split name, folder path, and CSV filename for raw source files."""

    for path, dir_lst, file_lst in os.walk(base_data_path):
        path_info = path.split("\\")
        if len(path_info) < 6:
            continue
        if "img" in path_info[5] or "decoupling" in path_info[5]:
            continue
        gas_name = path_info[-2]
        set_type = path_info[-1]
        if gas_name not in gas_name_array:
            continue
        for file_name in sorted(file_lst):
            if not file_name.lower().endswith(".csv"):
                continue
            yield gas_name, set_type, path, file_name


def _resolve_row_range(data_frame, gas_name):
    """Return the segmentable row range for one CSV file."""

    xdata = data_frame.iloc[:, 0]
    start_row = gas_dict[gas_name]
    end_row = min(start_row + 9400, xdata.shape[0])
    if xdata.shape[0] < 9600:
        start_row = 0
        end_row = min(start_row + 9400, xdata.shape[0])
    return start_row, end_row


def _prepare_output_dir(base_data_path, gas_name, set_type, graph_type, save_gray):
    """Create and return the output folder for one gas/split/graph-type combination."""

    output_graph_type = get_output_graph_type(graph_type, save_gray=save_gray)
    img_save_path = base_data_path + "\\" + gas_name + "\\" + set_type + output_graph_type
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    return img_save_path


def _save_segments_from_csv(
    csv_path,
    gas_name,
    img_save_path,
    size,
    img_size,
    matrix_builder,
    data_column="average",
    save_gray=False,
    file_index=1,
):
    """Transform one CSV into multiple graph images while keeping the legacy save layout."""

    data_frame = pd.read_csv(csv_path)
    start_row, end_row = _resolve_row_range(data_frame, gas_name)
    print(f"start row: {start_row}")
    ydata = np.asarray(data_frame[data_column], dtype=np.float32)
    print(ydata)

    ppm = 0
    for cur_row in range(start_row, end_row, size):
        if cur_row + size > end_row:
            continue
        ppm += 10
        segment = ydata[cur_row:cur_row + size]
        matrix = matrix_builder(segment, img_size)
        normalized_matrix = _normalize_matrix_to_uint8(matrix)
        final_img = build_output_image(normalized_matrix, img_size=img_size, save_gray=save_gray)

        ppm_dir = img_save_path + "\\" + str(ppm) + "_ppm"
        if not os.path.exists(ppm_dir):
            os.makedirs(ppm_dir)
        img_name = gas_name + "_" + str(ppm) + "ppm_" + str(file_index) + ".png"
        img_save_path_single = ppm_dir + "\\" + img_name
        cv2.imwrite(img_save_path_single, final_img)
        print(f"img {img_name} is saved in {img_save_path_single}")


def _create_graphs_from_average_data(
    base_data_path,
    size,
    img_size,
    graph_type,
    matrix_builder,
    data_column="average",
    save_gray=False,
):
    """Shared CSV-to-image conversion flow used by GASF, RP, and MTF."""

    folder_file_index = {}
    for gas_name, set_type, path, file_name in _iter_source_csv_files(base_data_path):
        csv_path = path + "\\" + file_name
        img_save_path = _prepare_output_dir(base_data_path, gas_name, set_type, graph_type, save_gray)
        file_index = folder_file_index.get(img_save_path, 0) + 1
        folder_file_index[img_save_path] = file_index
        _save_segments_from_csv(
            csv_path=csv_path,
            gas_name=gas_name,
            img_save_path=img_save_path,
            size=size,
            img_size=img_size,
            matrix_builder=matrix_builder,
            data_column=data_column,
            save_gray=save_gray,
            file_index=file_index,
        )


def create_MTF_from_average_data(
    base_data_path,
    size,
    img_size,
    n_bins,
    graph_type="_MTF_img",
    data_type="average",
    save_gray=False,
):
    """Convert response CSV files into Markov Transition Field images."""

    mtf = MarkovTransitionField(image_size=img_size, n_bins=n_bins, strategy="quantile")

    def matrix_builder(segment, current_img_size):
        data = np.asarray(segment, dtype=np.float32).reshape(1, -1)
        mtf_matrix = np.asarray(mtf.fit_transform(data))
        return mtf_matrix.reshape((current_img_size, current_img_size))

    _create_graphs_from_average_data(
        base_data_path=base_data_path,
        size=size,
        img_size=img_size,
        graph_type=graph_type,
        matrix_builder=matrix_builder,
        data_column=data_type,
        save_gray=save_gray,
    )


def create_RP_from_average_data(
    base_data_path,
    size,
    img_size,
    dimension,
    time_delay,
    graph_type="_RP_img",
    data_type="average",
    save_gray=False,
):
    """Convert response CSV files into Recurrence Plot images."""

    rp = RecurrencePlot(dimension=dimension, time_delay=time_delay)

    def matrix_builder(segment, _current_img_size):
        data = np.asarray(segment, dtype=np.float32).reshape(1, -1)
        rp_matrix = np.asarray(rp.transform(data))
        rp_size = rp_matrix.shape[2]
        return rp_matrix.reshape((rp_size, rp_size))

    _create_graphs_from_average_data(
        base_data_path=base_data_path,
        size=size,
        img_size=img_size,
        graph_type=graph_type,
        matrix_builder=matrix_builder,
        data_column=data_type,
        save_gray=save_gray,
    )


def create_GASF_from_average_data(
    base_data_path,
    size,
    img_size,
    graph_type="_GASF_img",
    data_type="average",
    save_gray=False,
):
    """Convert response CSV files into Gramian Angular Summation Field images."""

    def matrix_builder(segment, _current_img_size):
        return create_GASF(segment)

    _create_graphs_from_average_data(
        base_data_path=base_data_path,
        size=size,
        img_size=img_size,
        graph_type=graph_type,
        matrix_builder=matrix_builder,
        data_column=data_type,
        save_gray=save_gray,
    )


def create_CWT_from_average_data(
    base_data_path,
    size,
    img_size,
    scales,
    wavelet="morl",
    graph_type="_CWT_img",
    data_type="average",
    save_gray=False,
):
    """Convert response CSV files into Continuous Wavelet Transform images."""

    pywt = _import_pywt()
    if isinstance(scales, int):
        scales = np.arange(1, scales + 1)
    else:
        scales = np.asarray(scales)
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError("scales must be a positive integer or a non-empty one-dimensional sequence.")

    def matrix_builder(segment, _current_img_size):
        coefficients, _ = pywt.cwt(np.asarray(segment, dtype=np.float32), scales, wavelet)
        return np.abs(coefficients)

    _create_graphs_from_average_data(
        base_data_path=base_data_path,
        size=size,
        img_size=img_size,
        graph_type=graph_type,
        matrix_builder=matrix_builder,
        data_column=data_type,
        save_gray=save_gray,
    )


def create_STFT_from_average_data(
    base_data_path,
    size,
    img_size,
    fs=1.0,
    nperseg=64,
    noverlap=32,
    graph_type="_STFT_img",
    data_type="average",
    save_gray=False,
):
    """Convert response CSV files into Short-Time Fourier Transform images."""

    signal = _import_scipy_signal()
    if nperseg <= 0:
        raise ValueError("nperseg must be greater than 0.")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg.")

    def matrix_builder(segment, _current_img_size):
        _, _, stft_matrix = signal.stft(
            np.asarray(segment, dtype=np.float32),
            fs=fs,
            nperseg=min(int(nperseg), len(segment)),
            noverlap=min(int(noverlap), max(len(segment) - 1, 0), int(nperseg) - 1),
            boundary=None,
            padded=False,
        )
        return np.abs(stft_matrix)

    _create_graphs_from_average_data(
        base_data_path=base_data_path,
        size=size,
        img_size=img_size,
        graph_type=graph_type,
        matrix_builder=matrix_builder,
        data_column=data_type,
        save_gray=save_gray,
    )


def create_img_single(
    data_path,
    size,
    img_size,
    index,
    save_gray=False,
    graph_method="gasf",
    graph_type=None,
    data_column="average",
    n_bins=4,
    dimension=3,
    time_delay=3,
    scales=32,
    wavelet="morl",
    fs=1.0,
    nperseg=64,
    noverlap=32,
):
    """Generate graph images for one CSV file and save them by ppm segment."""

    path_info = data_path.split("\\")
    gas_name = path_info[-3]
    set_type = path_info[-2]

    graph_method = graph_method.lower()
    if graph_method == "gasf":
        if graph_type is None:
            graph_type = "_img"

        def matrix_builder(segment, _current_img_size):
            return create_GASF(segment)
    elif graph_method == "rp":
        if graph_type is None:
            graph_type = "_RP_img"
        rp = RecurrencePlot(dimension=dimension, time_delay=time_delay)

        def matrix_builder(segment, _current_img_size):
            data = np.asarray(segment, dtype=np.float32).reshape(1, -1)
            rp_matrix = np.asarray(rp.transform(data))
            rp_size = rp_matrix.shape[2]
            return rp_matrix.reshape((rp_size, rp_size))
    elif graph_method == "mtf":
        if graph_type is None:
            graph_type = "_MTF_img"
        mtf = MarkovTransitionField(image_size=img_size, n_bins=n_bins, strategy="quantile")

        def matrix_builder(segment, current_img_size):
            data = np.asarray(segment, dtype=np.float32).reshape(1, -1)
            mtf_matrix = np.asarray(mtf.fit_transform(data))
            return mtf_matrix.reshape((current_img_size, current_img_size))
    elif graph_method == "cwt":
        if graph_type is None:
            graph_type = "_CWT_img"
        pywt = _import_pywt()
        if isinstance(scales, int):
            scales = np.arange(1, scales + 1)
        else:
            scales = np.asarray(scales)

        def matrix_builder(segment, _current_img_size):
            coefficients, _ = pywt.cwt(np.asarray(segment, dtype=np.float32), scales, wavelet)
            return np.abs(coefficients)
    elif graph_method == "stft":
        if graph_type is None:
            graph_type = "_STFT_img"
        signal = _import_scipy_signal()

        def matrix_builder(segment, _current_img_size):
            _, _, stft_matrix = signal.stft(
                np.asarray(segment, dtype=np.float32),
                fs=fs,
                nperseg=min(int(nperseg), len(segment)),
                noverlap=min(int(noverlap), max(len(segment) - 1, 0), int(nperseg) - 1),
                boundary=None,
                padded=False,
            )
            return np.abs(stft_matrix)
    else:
        raise ValueError(
            "Unsupported graph_method: {}. Expected 'gasf', 'rp', 'mtf', 'cwt', or 'stft'.".format(graph_method)
        )

    img_save_path = base_data_path + "\\" + gas_name + "\\" + set_type + get_output_graph_type(graph_type, save_gray=save_gray)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    _save_segments_from_csv(
        csv_path=data_path,
        gas_name=gas_name,
        img_save_path=img_save_path,
        size=size,
        img_size=img_size,
        matrix_builder=matrix_builder,
        data_column=data_column,
        save_gray=save_gray,
        file_index=index,
    )


def create_GASF(data):
    """Build a GASF matrix from one normalized one-dimensional response segment."""

    normalized_data = normalization(data)
    normalized_data = np.clip(normalized_data, 0.0, 1.0)
    cosine_term = normalized_data[:, None] * normalized_data[None, :]
    sine_term = np.sqrt(1 - normalized_data[:, None] ** 2) * np.sqrt(1 - normalized_data[None, :] ** 2)
    return cosine_term - sine_term


def normalization(sequence):
    """Min-max normalize one sequence and return a numpy array."""

    sequence = np.asarray(sequence, dtype=np.float32).copy()
    list_max = float(np.max(sequence))
    list_min = float(np.min(sequence))
    if list_max == list_min:
        return np.zeros_like(sequence, dtype=np.float32)
    return (sequence - list_min) / (list_max - list_min)


if __name__ == "__main__":
    create_GASF_from_average_data(base_data_path, 940, 120, save_gray=False)
    # create_RP_from_average_data(base_data_path, 940, 120, 3, 3, save_gray=True)
    # create_MTF_from_average_data(base_data_path, 940, 120, 4, save_gray=True)
    # create_CWT_from_average_data(base_data_path, 940, 120, scales=32, wavelet="morl", save_gray=False)
    # create_STFT_from_average_data(base_data_path, 940, 120, fs=2.0, nperseg=64, noverlap=8, save_gray=False)
    # pass
