import torch
from torch import Tensor
from typing import Dict, List
from core.common.constants import *


class TensorInstance(object):
    def to_tensor(self):
        raise NotImplementedError()

    @property
    def empty_tensor(self):
        raise NotImplementedError


class ValueInstance(TensorInstance):
    def __init__(self, instance):
        self.instance = instance

    def to_tensor(self):
        tensor = torch.tensor(self.instance)
        return tensor

    @property
    def empty_tensor(self):
        raise NotImplementedError


class ListInstance(TensorInstance):
    def __init__(self, list_instances):
        self.list_instances = list_instances

    def to_tensor(self):
        if isinstance(self.list_instances[0], int) or \
                isinstance(self.list_instances[0], float) or \
                isinstance(self.list_instances[0], list):
            tensor = torch.stack([torch.tensor(instance) for instance in self.list_instances], dim=0)
        elif isinstance(self.list_instances[0], TensorInstance):
            tensor = torch.stack([instance.to_tensor() for instance in self.list_instances], dim=0)
        else:
            raise NotImplementedError
        return tensor

    def __len__(self):
        return len(self.list_instances)

    @property
    def empty_tensor(self):
        raise NotImplementedError


def batch_to_device(batch, device):
    if isinstance(batch, Tensor):
        batch = batch.to(device)
    elif isinstance(batch, Dict):
        for k, v in batch.items():
            if k != META_DATA:
                batch[k] = batch_to_device(v, device)
    elif isinstance(batch, List):
        for l in range(len(batch)):
            for k, v in batch[l].items():
                if k != META_DATA:
                    batch[l][k] = batch_to_device(v, device)
    else:
        raise NotImplementedError
    return batch

