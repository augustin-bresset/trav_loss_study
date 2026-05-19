import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from .lovasz_softmax import Lovasz_softmax
from .dice_loss import MultiLabelDice, DiceLoss

from .custom_losses import (
    ClassificationCriterion,
    CategoricalMSE,
    RegressionCriterion,
)


class CrossEntropy(nn.Module):
    def __init__(self, ignore_index=0, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, input, target):
        return self.ce(input, target)


LOSS_DICT = {
    "cross_entropy": CrossEntropy,
    "mse": nn.MSELoss,
    "multi_dice": MultiLabelDice,
    "dice": DiceLoss,
    "lovasz": Lovasz_softmax,
    "categorical_mse": CategoricalMSE,
}


class ClassificationCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        ignore_index,
        losses=["cross_entropy", "lovasz", "multi_dice", "categorical_mse"],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.loss_funcs = {}

        for loss in losses:
            print(f"Using {loss}")
            self.loss_funcs[loss] = LOSS_DICT[loss](
                num_classes=num_classes, ignore_index=ignore_index
            )

        self.N = len(losses)

    def forward(self, input, target):
        info_dict = {}
        loss = 0

        # self.ce(input, target.argmax(dim=1))
        # self.lovasz(input, target.long())
        # self.dice(input, target.long())

        for loss_name, loss_func in self.loss_funcs.items():
            # if loss_name == "cross_entropy":
            #     target_ = target.argmax(dim=1)
            if loss_name in ["lovasz", "multi_dice"]:
                target_ = target.long()
            else:
                target_ = target
            loss_val = loss_func(input, target_)
            loss += loss_val.mean()
            info_dict[loss_name] = loss_val.item()

        return loss / self.N, info_dict


class RegressionCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input.squeeze(), target.squeeze())
