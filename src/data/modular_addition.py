import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ModularAdditionDataset(Dataset):
    def __init__(self, p: int = 113, seed: int = 42, val_split: float = 0.3):
        self.p = p
        self.seed = seed
        self.val_split = val_split

        # Generate all p^2 pairs
        a = torch.arange(p, dtype=torch.long).unsqueeze(1).expand(p, p).reshape(-1)
        b = torch.arange(p, dtype=torch.long).unsqueeze(0).expand(p, p).reshape(-1)
        labels = (a + b) % p

        # inputs: [a, b, =] where = is token ID = p
        eq_token = torch.full_like(a, p)
        self.inputs = torch.stack([a, b, eq_token], dim=1)   # [p^2, 3]
        self.labels = labels                                  # [p^2]

    def __len__(self):
        return self.p * self.p

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def train_test_split(dataset: ModularAdditionDataset, val_split: float, seed: int):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


class DataModule:
    def __init__(
        self,
        p: int = 113,
        val_split: float = 0.3,
        seed: int = 42,
        batch_size: int = 12769,
        num_workers: int = 0,
    ):
        self.p = p
        self.val_split = val_split
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        full_dataset = ModularAdditionDataset(p=p, seed=seed, val_split=val_split)
        self.train_dataset, self.val_dataset = train_test_split(
            full_dataset, val_split, seed
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
