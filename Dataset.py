import os
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image as im
from sklearn.model_selection import KFold
import random

base_data_path = r"D:\E-nose_DataSet\Gas_Recognition_DataSet\gas_response_data"
gas_name_array = ["1-butanol", "acetone", "benzaldehyde", "butyl acetate", "dimethylbenzene"]
gas_label = {"1-butanol": 0, "acetone": 1, "benzaldehyde": 2, "butyl acetate": 3, "dimethylbenzene": 4}
gas_display_name = {
    "1-butanol": "1BA",
    "acetone": "AC",
    "benzaldehyde": "BZ",
    "butyl acetate": "BAC",
    "dimethylbenzene": "DMB",
    "methanol": "MT",
}
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize every generated graph image to the model input size.
    transforms.ToTensor(),
    # transforms.Normalize(0.485, 0.229)
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class MyDataset(Dataset):
    """Classification dataset that returns graph images and gas class labels."""

    def __init__(self, base_data_path, set_type, transform=None, gas_name_array=None, gas_label=None, img_type="img"):
        """Collect image paths under each gas folder and assign class labels."""

        img_list = []
        label_list = []
        for gas_name in gas_name_array:
            set_data_path = base_data_path + "\\" + gas_name + "\\" + set_type + "_set_" + img_type
            print(set_data_path)
            for path, dir_lst, file_lst in os.walk(set_data_path):
                for file_name in file_lst:
                    ppm = file_name.split('_')[1]
                    ppm_dir = ppm[:-3] + "_ppm"
                    img_data_path = set_data_path + "\\" + ppm_dir + "\\" + file_name
                    img_list.append(img_data_path)
                    label_list.append(gas_label[gas_name])
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        print(len(img_list))
        print(len(label_list))

    def __len__(self):
        """Return the number of images in the selected split."""

        return len(self.img_list)

    def __getitem__(self, index):
        """Load one image, apply transforms, and return its class label."""

        img = im.open(self.img_list[index])
        # img = cv2.imread(self.img_list[index])
        img = self.transform(img)
        label = self.label_list[index]

        return img, label


class MyDataset_regression(Dataset):
    """Regression dataset that returns graph images and ppm concentration labels."""

    def __init__(self, base_data_path, set_type, transform=None, gas_name_array=None, gas_label=None, img_type="img"):
        """Collect image paths and parse ppm values from file names."""

        img_list = []
        label_list = []
        if set_type not in ["train", "test"]:
            raise ValueError("set_type must be 'train' or 'test'")

        for gas_name in gas_name_array:
            set_data_path = base_data_path + "\\" + gas_name + "\\" + set_type + "_set_" + img_type
            for path, dir_lst, file_lst in os.walk(set_data_path):
                for file_name in file_lst:
                    ppm = file_name.split('_')[1]
                    ppm_dir = ppm[:-3] + "_ppm"
                    img_data_path = set_data_path + "\\" + ppm_dir + "\\" + file_name
                    img_list.append(img_data_path)
                    label_list.append(float(ppm[:-3]))
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        print(len(img_list))
        print(len(label_list))

    def __len__(self):
        """Return the number of images in the selected split."""

        return len(self.img_list)

    def __getitem__(self, index):
        """Load one image, apply transforms, and return its ppm label."""

        img = im.open(self.img_list[index])
        # img = cv2.imread(self.img_list[index])
        img = self.transform(img)
        label = self.label_list[index]
        return img, label


class GeneratedGasImageDataset(Dataset):
    """Classification dataset for the generated gas-response images under gas_response_data2."""

    def __init__(self, base_data_path, set_type, transform=None):
        """Discover gas folders automatically and collect image paths for one split."""

        if set_type not in ["train", "test"]:
            raise ValueError("set_type must be 'train' or 'test'")

        base_path = Path(base_data_path)
        if not base_path.exists():
            raise FileNotFoundError("base_data_path does not exist: {}".format(base_path))

        candidate_dirs = []
        for item in sorted(base_path.iterdir(), key=lambda path: path.name.lower()):
            if not item.is_dir():
                continue
            if not (item / "raw_data").exists():
                continue
            if not ((item / "train_set_img").exists() or (item / "test_set_img").exists()):
                continue
            candidate_dirs.append(item)

        if not candidate_dirs:
            raise FileNotFoundError(
                "No valid gas image folders were found under: {}".format(base_path)
            )

        self.class_names = [item.name for item in candidate_dirs]
        self.class_display_names = [
            gas_display_name.get(class_name, class_name) for class_name in self.class_names
        ]
        self.class_to_label = {
            class_name: class_index for class_index, class_name in enumerate(self.class_names)
        }

        split_dir_name = "{}_set_img".format(set_type)
        img_list = []
        label_list = []
        for gas_dir in candidate_dirs:
            image_dir = gas_dir / split_dir_name
            if not image_dir.exists():
                continue
            image_files = sorted(
                [
                    image_path for image_path in image_dir.iterdir()
                    if image_path.is_file() and image_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
                ],
                key=self._image_sort_key,
            )
            for image_path in image_files:
                img_list.append(str(image_path))
                label_list.append(self.class_to_label[gas_dir.name])

        if not img_list:
            raise FileNotFoundError(
                "No image files were found for split '{}' under: {}".format(set_type, base_path)
            )

        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        print("[GeneratedGasImageDataset] split={}, class_count={}, sample_count={}".format(
            set_type,
            len(self.class_names),
            len(self.img_list),
        ))

    @staticmethod
    def _image_sort_key(image_path):
        """Sort numbered files numerically first, then fall back to the file name."""

        stem = image_path.stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, image_path.name.lower())

    def __len__(self):
        """Return the number of generated images in the selected split."""

        return len(self.img_list)

    def __getitem__(self, index):
        """Load one generated image, apply transforms, and return its class label."""

        img = im.open(self.img_list[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.label_list[index]
        return img, label


if __name__ == "__main__":
    # data_path = r"D:\E-nose_DataSet\Work2_12VOCs_DataSet\gas_response_data\1-butanol\train_set_img\10_ppm\1-butanol_10ppm_1.png"
    # img = im.open(data_path)
    # img.show()
    # X = np.arange(150)
    # kf = KFold(n_splits=5, shuffle=True)
    # for train_index, test_index in kf.split(X):  # Split data with KFold.
    #     # print('train_index:%s , test_index: %s ' % (train_index, test_index))
    #     dataset_train = MyDataset_corssValid(base_data_path, train_index, None, gas_name_array, gas_label)
    #     dataset_test = MyDataset_corssValid(base_data_path, test_index, None, gas_name_array, gas_label)

    List = random.sample(range(0, 120), 10)
