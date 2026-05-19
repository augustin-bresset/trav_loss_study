import os
import csv
import yaml
import argparse

def _build_label_cfg_goose(dataset_dir):
    """
    GOOSE_3D
    │   ├── test
    │   │   └── gooseEx_3d_test
    │   ├── train
    │   │   ├── CHANGELOG
    │   │   ├── goose_label_mapping.csv
    │   │   ├── height_labels
    │   │   ├── labels
    │   │   ├── LICENSE
    │   │   └── lidar
    │   └── val
    │       └── gooseEx_3d_val
    """

    goose_label_path  = os.path.join(dataset_dir, "GOOSE_3D", "train", "goose_label_mapping.csv")

    if not os.path.exists(goose_label_path):
        raise FileNotFoundError(f"Goose label file not found at {goose_label_path}")

    semantic_map = dict()
    color_map = dict()

    with open(goose_label_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['label_key'])
            class_name = row['class_name']
            color = row['hex']
            semantic_map[class_id] = class_name
            color_map[class_id] = color

    return {
        "semantic_map": semantic_map,
        "num_classes": len(semantic_map),
        "ignore_index": 0,
        "color_map": color_map,
    }

def _build_label_cfg_rellis(dataset_dir):
    """
    RELLIS-3D label config builder from ontology.csv
    """

    ontology_path = os.path.join(
        dataset_dir,
        "Rellis_3D_ontology",
        "ontology.csv"
    )

    if not os.path.exists(ontology_path):
        raise FileNotFoundError(f"RELLIS ontology CSV not found at {ontology_path}")

    semantic_map = {}
    color_map = {}

    with open(ontology_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            class_id = int(row["output_value"])
            class_name = row["class_name"]
            color_hex = row["display_color"]

            # parse hex color -> RGB
            color_hex = color_hex.lstrip("#")
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)

            semantic_map[class_id] = class_name
            color_map[class_id] = [r, g, b]

    # ensure sorted consistency (important for reproducibility)
    semantic_map = dict(sorted(semantic_map.items()))
    color_map = dict(sorted(color_map.items()))

    return {
        "semantic_map": semantic_map,
        "num_classes": len(semantic_map),
        "ignore_index": 0,
        "color_map": color_map,
    }

def ___build_label_cfg_rellis(dataset_dir):
    """
    RELLIS-3D
    ontology.yaml structure:
    [
        {class_id: class_name, ...},
        {class_id: [r,g,b], ...}
    ]
    """

    ontology_path = os.path.join(
        dataset_dir,
        "Rellis_3D_ontology",
        "ontology.yaml"
    )

    if not os.path.exists(ontology_path):
        raise FileNotFoundError(f"RELLIS ontology file not found at {ontology_path}")

    with open(ontology_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("Unexpected RELLIS ontology format (expected 2 YAML blocks)")

    semantic_map_raw = data[0]
    color_map_raw = data[1]

    # clean + ensure int keys
    semantic_map = {int(k): v for k, v in semantic_map_raw.items()}
    color_map = {int(k): v for k, v in color_map_raw.items()}

    return {
        "semantic_map": semantic_map,
        "num_classes": len(semantic_map),
        "ignore_index": 0,
        "color_map": color_map,
    }

PARSERS = {
    "goose": _build_label_cfg_goose,
    "rellis": _build_label_cfg_rellis,
}


def build_label_cfg(dataset_name, dataset_dir, output_dir="resources"):
    """ Build a label config yaml file ready to be used by training.
    This function will groups different function used to convert specific label files into 
    this generic label config file.
    """


    if not dataset_name in PARSERS.keys():
        raise ValueError(f"{dataset_name} not supported, go make one into src/preprocessing")
    
    label_builder = PARSERS[dataset_name]
    metadata = label_builder(dataset_dir) 
    
    output_file = os.path.join(output_dir, f"{dataset_name}_cfg.yaml")

    with open(output_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Domain Invariance")
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the Dataset"
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()
    
    build_label_cfg(
        dataset_name=args.name,
        dataset_dir=args.path
        )
