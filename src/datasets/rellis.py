import importlib
import os
import logging
import torch
import torch.nn.functional as F
from torch._C import Value
from torch_geometric.data import Data, Dataset
import logging
from pathlib import Path
from glob import glob
import pandas as pd

# Basic libs
import numpy as np
import yaml

from .dataset_mapping import go_to_split, find_dataset_root, find_raw_data_dirs

from .utils import heights_to_bins



class Rellis3D(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        max_samples=5000,
        remap_cfg=None,
        max_rad=50,
        transform=None,
        feat_dup=True,
    ):
        super().__init__(root_dir, transform, None)

        self.root_dir = root_dir
        self.split = split
        self.max_samples = max_samples
        self.max_rad = max_rad
        self.feat_dup = feat_dup
        self.remap_cfg = remap_cfg
        self.num_auxiliary_classes_ = 10

        if self.feat_dup:
            self.feat_dim = 4
        else:
            self.feat_dim = 1

        self.ontology_df = pd.read_csv(
            os.path.join(self.root_dir, "Rellis_3D_ontology", "ontology.csv")
        )

        self.num_classes_ = np.max(self.ontology_df["output_value"]) + 1
        self.ignore_indices = list(
            set(range(self.num_classes_)) - set(self.ontology_df["output_value"])
        )

        self.split_file = os.path.join(self.root_dir, f"pt_{self.split}.lst")

        self.og_label_map = dict(
            zip(self.ontology_df["output_value"], self.ontology_df["class_name"])
        )

        self.og_label_to_label = {v: v for v in self.ontology_df["output_value"]}

        self.remap_dict = None

        if self.remap_cfg is not None:
            self.remap_cfg = os.path.join(self.root_dir, remap_cfg)
            with open(self.remap_cfg, "r") as stream:
                remap_data = yaml.safe_load(stream)
                self.remap_dict = remap_data["learning_map"]
            self.og_label_to_label = {
                k: self.remap_dict[v] for k, v in self.og_label_to_label.items()
            }
            self.num_classes_ = len(set(self.remap_dict.values()))
            print(f"RELLIS Number of classes: {self.num_classes_}")

        self.label_to_og_label = {v: k for k, v in self.og_label_to_label.items()}

        self.file_name_df = pd.read_csv(self.split_file, header=None, sep=" ")
        self.file_name_df.columns = ["file_name", "label"]

        self.file_name_df["height_label"] = [
            x.replace("os1_cloud_node_kitti_bin", "height_labels").replace(
                ".bin", ".height_label"
            )
            for x in self.file_name_df["file_name"]
        ]

        if self.max_samples is not None and len(self.file_name_df) > self.max_samples:
            # if len(self.file_name_df) > self.max_samples:
            self.file_name_df = self.file_name_df.sample(self.max_samples)

        self.file_name_df["file_name"] = [
            os.path.join(self.root_dir, "Rellis-3D", f) for f in self.file_name_df["file_name"]
        ]
        self.file_name_df["label"] = [
            os.path.join(self.root_dir, "Rellis-3D", f) for f in self.file_name_df["label"]
        ]

        self.file_name_df["height_label"] = [
            os.path.join(self.root_dir, "Rellis-3D", f) for f in self.file_name_df["height_label"]
        ]

        self.id_to_file_name = dict(
            zip(range(len(self.file_name_df)), self.file_name_df["file_name"])
        )
        self.id_to_label = dict(
            zip(range(len(self.file_name_df)), self.file_name_df["label"])
        )

        self.id_to_height_label = dict(
            zip(range(len(self.file_name_df)), self.file_name_df["height_label"])
        )

    def __len__(self):
        return len(self.file_name_df)

    def get_filenames(self):
        return list(self.file_name_df["file_name"])

    def get_bin_dir(self):
        return "os1_cloud_node_kitti_bin"

    def len(self):
        return len(self.file_name_df)

    def _download(self):  # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        file_name = self.id_to_file_name[idx]
        label_file = self.id_to_label[idx]
        height_label_file = self.id_to_height_label[idx]

        pos = np.fromfile(file_name, dtype=np.float32).reshape((-1, 4))
        features = pos[:, 3:]
        coords = pos[:, :3]

        labels = np.fromfile(label_file, dtype=np.int32).reshape(-1)
        labels = labels & 0xFFFF  # semantic label in lower half

        height_labels = np.float32(np.fromfile(height_label_file))
        # print(coords.shape, labels.shape, height_labels.shape)
        height_labels = heights_to_bins(height_labels)

        mask = np.linalg.norm(coords, axis=1) < self.max_rad
        coords = coords[mask]
        features = features[mask]

        labels = labels[mask]
        height_labels = height_labels[mask]

        if self.remap_cfg is not None:
            labels = [self.og_label_to_label[l] for l in labels]

        coords = torch.tensor(coords, dtype=torch.float)
        features = torch.tensor(features, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        height_labels = torch.tensor(height_labels, dtype=torch.long)

        x = torch.ones((coords.shape[0], 1), dtype=torch.float)
        if self.feat_dup:
            features = torch.cat([features, coords], dim=1)
        return Data(
            x=features,
            intensities=features,
            pos=coords,
            y=labels,
            shape_id=idx,
            pcd_file=file_name,
            height_labels=height_labels,
            label_file=label_file,

        )

