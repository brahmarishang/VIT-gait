
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from vitg.models.vitg import VITG
from vitg.data.datasets import make_loaders

@dataclass
class TrainConfig:
    train_dir: str
    val_dir: str
    img_size: int = 128
    patch_size: int = 32
    embed_dim: int = 64
    depth: int = 16
    num_heads: int = 8
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-6
    max_epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def run_training(cfg: TrainConfig):
    train_loader, val_loader, n_classes = make_loaders(
        cfg.train_dir, cfg.val_dir, size=(cfg.img_size, cfg.img_size), batch_size=cfg.batch_size
    )
    model = VITG(
        img_size=cfg.img_size, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim,
        depth=cfg.depth, num_heads=cfg.num_heads, num_classes=n_classes
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_acc, best_state = 0.0, None
    for epoch in range(cfg.max_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step()
        print(f"Epoch {epoch+1:03d}/{cfg.max_epochs} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = model.state_dict()

    if best_state is not None:
        import pathlib
        out = pathlib.Path("artifacts")
        out.mkdir(exist_ok=True)
        torch.save(best_state, out / "vitg_best.pt")
        print(f"Saved best model with acc={best_acc:.4f} to {out/'vitg_best.pt'}")
    return best_acc
