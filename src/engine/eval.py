
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from vitg.models.vitg import VITG
from vitg.data.datasets import GEIDataset

@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, data_dir: str, img_size: int = 128, patch_size: int = 32,
                        embed_dim: int = 64, depth: int = 16, num_heads: int = 8,
                        batch_size: int = 32, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = GEIDataset(data_dir, size=(img_size, img_size), augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    n_classes = len(set(lbl for _, lbl in ds.samples))

    model = VITG(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                 depth=depth, num_heads=num_heads, num_classes=n_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    return acc, prec, rec, f1
