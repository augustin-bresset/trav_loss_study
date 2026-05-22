import torch
from torchsparse import SparseTensor


def to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)

    if isinstance(data, SparseTensor):
        return data.to(device)

    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}

    if isinstance(data, list):
        return [to_device(v, device) for v in data]

    if isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)

    return data