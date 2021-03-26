import torch
from torch.utils.data import Dataset

from core.common.constants import META_DATA


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class CustomCollate:
    @staticmethod
    def collate(instances):
        batch = {}
        for instance in instances:
            for k, v in instance.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        for k, instances in batch.items():
            if k != META_DATA:
                batch[k] = torch.stack([instance.to_tensor() for instance in instances], dim=0)
        return batch
