import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset
from wilds.common.grouper import CombinatorialGrouper


class Camelyon17UnlabeledDataset(WILDSUnlabeledDataset):
    """
    Unlabeled Camelyon17 dataset.
    This is a modified version of the original CAMELYON17 dataset. We took the patches not in the
    lesion annotations for unlabeled data.

    Supported `split_scheme`:
        'official'

    Input (x):
        96x96 image patches extracted from histopathology slides.

    Metadata:
        Each patch is annotated with the ID of the hospital it came from (integer from 0 to 4)
        and the slide it came from (integer from 0 to 49).

    Website:
        https://camelyon17.grand-challenge.org/

    Original publication:
        @article{bandi2018detection,
          title={From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge},
          author={Bandi, Peter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Bejnordi, Babak Ehteshami and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and others},
          journal={IEEE transactions on medical imaging},
          volume={38},
          number={2},
          pages={550--560},
          year={2018},
          publisher={IEEE}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    _dataset_name = "camelyon17_unlabeled"
    # TODO: update with the final dataset bundle uuid and size -Tony
    _versions_dict = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/",
            "compressed_size": 10_658_709_504,
        }
    }

    def __init__(
        self, version=None, root_dir="data", download=False, split_scheme="official"
    ):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._original_resolution = (96, 96)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, "metadata.csv"),
            index_col=0,
            dtype={"patient": "str"},
        )

        # Get filenames
        self._input_array = [
            f"patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"
            for patient, node, x, y in self._metadata_df.loc[
                :, ["patient", "node", "x_coord", "y_coord"]
            ].itertuples(index=False, name=None)
        ]

        # Extract splits
        # Note that the hospital numbering here is different from what's in the paper,
        # where to avoid confusing readers we used a 1-indexed scheme and just labeled the test hospital as 5.
        # Here, the numbers are 0-indexed.
        test_center = 2
        val_center = 1

        self._split_scheme = split_scheme
        if self._split_scheme == "official":
            self._split_dict = {
                "train_unlabeled": 10,
                "val_unlabeled": 11,
                "test_unlabeled": 12,
            }
            self._split_names = {
                "train_unlabeled": "Unlabeled Train",
                "val_unlabeled": "Unlabeled Validation",
                "test_unlabeled": "Unlabeled Test",
            }
        else:
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")

        centers = self._metadata_df["center"].values.astype("long")
        num_centers = int(np.max(centers)) + 1
        val_center_mask = self._metadata_df["center"] == val_center
        test_center_mask = self._metadata_df["center"] == test_center
        self._metadata_df.loc[val_center_mask, "split"] = self.split_dict["val_unlabeled"]
        self._metadata_df.loc[test_center_mask, "split"] = self.split_dict["test_unlabeled"]
        self._split_array = self._metadata_df["split"].values

        self._y_array = torch.LongTensor(self._metadata_df['tumor'].values)
        self._metadata_array = torch.stack(
            (
                torch.LongTensor(centers),
                torch.LongTensor(self._metadata_df["slide"].values),
                self._y_array,
            ),
            dim=1,
        )
        self._metadata_fields = ["hospital", "slide", "y"]

        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=["slide"]
        )

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = os.path.join(self.data_dir, self._input_array[idx])
        x = Image.open(img_filename).convert("RGB")
        return x