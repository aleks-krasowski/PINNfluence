import torch

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, return_zeroes=False):
        self.x = torch.tensor(x, requires_grad=True)
        self.return_zeroes = return_zeroes

    def __getitem__(self, idx):
        if self.return_zeroes:
            return self.x[idx], torch.zeros_like(self.x[idx, 0:1], requires_grad=True)
        return self.x[idx]

    def __len__(self):
        return len(self.x)