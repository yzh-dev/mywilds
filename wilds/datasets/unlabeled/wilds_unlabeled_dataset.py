import os

import torch
import numpy as np

from wilds.datasets.wilds_dataset import WILDSDataset


class WILDSUnlabeledDataset(WILDSDataset):
    """
    Shared dataset class for all unlabeled WILDS datasets.
    Each data point in the dataset is an (x, metadata) tuple, where:
    - x is the input features
    - metadata is a vector of relevant information, e.g., domain.
    """

    DEFAULT_SPLITS = {
        "train_unlabeled": 10,
        "val_unlabeled": 11,
        "test_unlabeled": 12,
        "extra_unlabeled": 13,
    }
    DEFAULT_SPLIT_NAMES = {
        "train_unlabeled": "Unlabeled Train",
        "val_unlabeled": "Unlabeled Validation",
        "test_unlabeled": "Unlabeled Test",
        "extra_unlabeled": "Unlabeled Extra",
    }

    _UNSUPPORTED_FUNCTIONALITY_ERROR = "Not supported - no labels available."

    def __len__(self):
        return len(self.metadata_array)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        metadata = self.metadata_array[idx]
        return x, metadata

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        if frac < 1.0:
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return WILDSUnlabeledSubset(self, split_idx, transform)

    def check_init(self):
        """
        Convenience function to check that the WILDSDataset is properly configured.
        """
        required_attrs = [
            "_dataset_name",
            "_data_dir",
            "_split_scheme",
            "_split_array",
            "_metadata_fields",
            "_metadata_array",
        ]
        for attr_name in required_attrs:
            assert hasattr(
                self, attr_name
            ), f"WILDSUnlabeledDataset is missing {attr_name}."

        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        # Check splits
        assert self.split_dict.keys() == self.split_names.keys()

        # Check that required arrays are Tensors
        assert isinstance(
            self.metadata_array, torch.Tensor
        ), "metadata_array must be a torch.Tensor"

        # Check that dimensions match
        assert len(self.split_array) == len(self.metadata_array)

        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]

    def initialize_data_dir(self, root_dir, download):
        self.check_version()

        os.makedirs(root_dir, exist_ok=True)
        dataset_name = f"{self.dataset_name}_v{self.version}"
        data_dir = os.path.join(root_dir, dataset_name)
        version_file = os.path.join(data_dir, f"UNLABELED_RELEASE_v{self.version}.txt")

        # If the dataset has an equivalent dataset, check if the equivalent dataset already exists
        # at the root directory. If it does, don't download and return the equivalent dataset path.
        version_dict = self.versions_dict[self.version]
        if "equivalent_dataset" in version_dict:
            equivalent_dataset_dir = os.path.join(
                root_dir, version_dict["equivalent_dataset"]
            )
            if os.path.exists(equivalent_dataset_dir):
                return equivalent_dataset_dir

        # If the dataset exists at root_dir, then don't download.
        if self.dataset_exists_locally(data_dir, version_file):
            return data_dir

        # Proceed with downloading.
        self.download_dataset(data_dir, download)
        return data_dir

    def eval(self, y_pred, y_true, metadata):
        raise AttributeError(WILDSUnlabeledDataset._UNSUPPORTED_FUNCTIONALITY_ERROR)

    @property
    def y_array(self):
        raise AttributeError(WILDSUnlabeledDataset._UNSUPPORTED_FUNCTIONALITY_ERROR)

    @property
    def y_size(self):
        raise AttributeError(WILDSUnlabeledDataset._UNSUPPORTED_FUNCTIONALITY_ERROR)


class WILDSUnlabeledSubset(WILDSUnlabeledDataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = [
            "_dataset_name",
            "_data_dir",
            "_collate",
            "_split_scheme",
            "_split_dict",
            "_split_names",
            "_metadata_fields",
            "_metadata_map",
        ]
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform

    def __getitem__(self, idx):
        x, metadata = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, metadata

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]
