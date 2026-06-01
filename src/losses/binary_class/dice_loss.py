import torch
import torch.nn as nn
import unittest


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, probs, targets):
        """
        Calculates the binary dice loss defined as `1 - 2 * p.y / (p+y+1)`.
        """
        intersection = torch.sum(probs * targets, dim=1)
        union = torch.sum(probs + targets, dim=1)
        dice_loss = 1 - 2 * intersection / (union + 1)

        return dice_loss.mean()


class MultiLabelDice(nn.Module):
    def __init__(self, num_classes, ignore_index=None):
        super(MultiLabelDice, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        smooth = 1e-6

        probs = logits  # [N, C]

        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)  # [N, C]

        # Sum over points (dim=0) per-class stats [C]
        intersection = torch.sum(probs * targets_onehot, dim=0)
        union = torch.sum(probs + targets_onehot, dim=0)

        dice = (2 * intersection + smooth) / (union + smooth)  # [C]

        if self.ignore_index is not None:
            mask = torch.ones(self.num_classes, dtype=torch.bool, device=logits.device)
            mask[self.ignore_index] = False
            dice = dice[mask]

        return 1 - dice.mean()


# Unit tests
class TestMultiLabelDice(unittest.TestCase):
    def setUp(self):
        self.num_classes = 5
        self.ignore_classes = [0, 3]
        self.loss_fn = MultiLabelDice(self.num_classes, self.ignore_classes)

    def test_forward(self):
        logits = torch.randn(2, self.num_classes, 32, 32)
        targets = torch.randint(0, self.num_classes, (2, 32, 32))

        dice_loss = self.loss_fn(logits, targets)

        self.assertEqual(dice_loss.shape, torch.Size([]))
        self.assertTrue(dice_loss >= 0)

    def test_forward_with_ignore_classes(self):
        logits = torch.randn(2, self.num_classes, 32, 32)
        targets = torch.randint(0, self.num_classes, (2, 32, 32))

        # Set some target pixels to ignore classes
        targets[targets == self.ignore_classes[0]] = self.ignore_classes[1]

        dice_loss = self.loss_fn(logits, targets)

        self.assertEqual(dice_loss.shape, torch.Size([]))
        self.assertTrue(dice_loss >= 0)

    def test_forward_with_empty_ignore_classes(self):
        logits = torch.randn(2, self.num_classes, 32, 32)
        targets = torch.randint(0, self.num_classes, (2, 32, 32))

        # Set some target pixels to ignore classes
        targets[targets == self.ignore_classes[0]] = self.ignore_classes[0]

        dice_loss = self.loss_fn(logits, targets)

        self.assertEqual(dice_loss.shape, torch.Size([]))
        self.assertTrue(dice_loss >= 0)


if __name__ == "__main__":
    unittest.main()
