"""CLI entry point: python -m src.visualization [options]"""

import argparse
import yaml

from src.datasets import Goose3D, Rellis3D, SemanticKITTI, Outback
from src.visualization.viewer import DatasetViewer


DATASET_CLASSES = {
    "goose": Goose3D,
    "rellis": Rellis3D,
    "kitti": SemanticKITTI,
    "outback": Outback,
}

DATASET_KWARGS = {
    "goose":   lambda a: dict(root_dir=a.root, split=a.split, max_samples=a.max_samples),
    "rellis":  lambda a: dict(root_dir=a.root, split=a.split, max_samples=a.max_samples),
    "kitti":   lambda a: dict(root=a.root, split=a.split),
    "outback": lambda a: dict(root_dir=a.root, split=a.split, max_samples=a.max_samples),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive LiDAR dataset viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.visualization --dataset goose  --root data/GOOSE  --cfg resources/goose_cfg.yaml
  python -m src.visualization --dataset rellis --root data/RELLIS  --cfg resources/rellis_cfg.yaml
  python -m src.visualization --dataset kitti  --root data/KITTI   --split val
        """,
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CLASSES),
                        help="Dataset name")
    parser.add_argument("--root", required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--split", default="train",
                        help="Dataset split (train / val / test). Default: train")
    parser.add_argument("--cfg", default=None,
                        help="Path to label config yaml (e.g. resources/goose_cfg.yaml). "
                             "Auto-detected from resources/<dataset>_cfg.yaml if omitted.")
    parser.add_argument("--idx", type=int, default=0,
                        help="Starting frame index. Default: 0")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap dataset size (useful for large datasets)")
    args = parser.parse_args()

    # Build dataset
    cls = DATASET_CLASSES[args.dataset]
    kwargs = DATASET_KWARGS[args.dataset](args)
    print(f"[viz] Loading {args.dataset.upper()} split='{args.split}' from {args.root} …")
    dataset = cls(**kwargs)
    print(f"[viz] {len(dataset)} frames loaded.")

    # Load label config
    label_cfg = None
    cfg_path = args.cfg or f"resources/{args.dataset}_cfg.yaml"
    try:
        with open(cfg_path) as f:
            label_cfg = yaml.safe_load(f)
        print(f"[viz] Label config loaded from {cfg_path}")
    except FileNotFoundError:
        print(f"[viz] No label config found at {cfg_path} — using auto colors.")

    DatasetViewer.launch(dataset, label_cfg, start_idx=args.idx)


if __name__ == "__main__":
    main()
