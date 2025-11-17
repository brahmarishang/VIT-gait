
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class GEIDataset(Dataset):
    """Generic GEI folder dataset: root/class_x/*.png -> (tensor, label)."""
    def __init__(self, root: str, size: Tuple[int, int] = (128, 128), augment: bool = False) -> None:
        self.root = Path(root)
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: List[Tuple[Path, int]] = []
        for c in classes:
            for p in (self.root / c).glob("*.png"):
                self.samples.append((p, self.class_to_idx[c]))

        tfm_list = [T.Grayscale(num_output_channels=1), T.Resize(size)]
        if augment:
            tfm_list.extend([
                T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05)),
                T.GaussianBlur(kernel_size=3, sigma=(0.0, 0.8)),
            ])
        tfm_list.append(T.ToTensor())
        self.transform = T.Compose(tfm_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("L")
        x = self.transform(img)
        return x, y


def make_loaders(train_dir: str, val_dir: str, size: Tuple[int, int] = (128, 128),
                 batch_size: int = 16, num_workers: int = 2):
    train_ds = GEIDataset(train_dir, size=size, augment=True)
    val_ds = GEIDataset(val_dir, size=size, augment=False)
    n_classes = len(set(lbl for _, lbl in train_ds.samples))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, n_classes
