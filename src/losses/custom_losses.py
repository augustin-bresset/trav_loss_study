import torch
import torch.nn as nn

from .lovasz_softmax import Lovasz_softmax
from .dice_loss import MultiLabelDice, DiceLoss


class ClassificationCriterion(nn.Module):
    def __init__(self, num_classes, ignore_index, return_individual=False):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.lovasz = Lovasz_softmax()
        self.dice = MultiLabelDice(num_classes, ignore_classes=[self.ignore_index])
        self.return_individual = return_individual

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        lovasz_loss = self.lovasz(input, target.long())
        dice_loss = self.dice(input, target.long())
        loss = (ce_loss + lovasz_loss + dice_loss) / 3
        if self.return_individual:
            return loss, (ce_loss.item(), lovasz_loss.item(), dice_loss.item())
        return loss


class BinaryClassificationCriterion(nn.Module):
    def __init__(self, return_individual=False):
        super().__init__()
        self.ce = nn.BCEWithLogitsLoss()
        self.lovasz = Lovasz_softmax()
        self.dice = DiceLoss()
        self.return_individual = return_individual

    def forward(self, input, target):
        ce_loss = self.ce(input, target.float())
        lovasz_loss = self.lovasz(input, target.long())
        dice_loss = self.dice(input, target.long())
        loss = (ce_loss + lovasz_loss + dice_loss) / 3
        if self.return_individual:
            return loss, (ce_loss.item(), lovasz_loss.item(), dice_loss.item())
        return loss


class CategoricalMSE(nn.Module):
    def __init__(self, num_classes, ignore_index=0, device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_vec = torch.arange(num_classes).to(torch.float32).to(device)

        # self.class_vec = self.class_vec[self.class_vec != ignore_index]
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        target = target.float()
        assert (
            input.shape[-1] == self.class_vec.T.shape[0]
        ), f"{input.shape} != {self.class_vec.shape}"
        input = input @ self.class_vec.T
        assert input.shape == target.shape, f"{input.shape} != {target.shape}"
        mask = target == self.ignore_index if self.ignore_index is not None else None
        # assert (
        #     input[~mask].shape == target[~mask].shape
        # ), f"{input[~mask].shape} != {target[~mask].shape}"
        try:
            if mask is None:
                return self.mse(input, target)
            else:
                return self.mse(input[~mask], target[~mask])
        except:
            print(input.shape, target.shape)
            if mask is None:
                return self.mse(input, target)
            else:

                return self.mse(input[~mask], target[~mask])


class RegressionCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        # print(input.shape, target.shape)
        # print(type(input), type(target))
        return self.mse(input.squeeze(), target.squeeze())
