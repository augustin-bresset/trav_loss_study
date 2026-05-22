import numpy as np
import os
import argparse
import yaml
import torch
import random

from src.core import Trainer


def get_args(config_file):
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)
    return argparse.Namespace(**config_dict)


def seed_everything(seed: int):
    """fix the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  #
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="Domain Invariance")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "predict"],
        help="Mode: train or test",
    )
    args_ = parser.parse_args()
    args = get_args(args_.config)

    if hasattr(args, "predictions_only"):
        args.predictions_only = (
            args.predictions_only if args.predictions_only is not None else False
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = f"cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")
    # exit()

    if hasattr(args, "seed"):
        args.seed = args.seed if args.seed is not None else 42
    else:
        args.seed = 42

    # seed_everything(args.seed)

    trainer = Trainer(args)
    if args_.mode == "train":
        trainer.train()
    elif args_.mode == "predict":
        trainer.predict()
    else:
        trainer.test()


if __name__ == "__main__":
    main()
