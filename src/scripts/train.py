
import argparse
from vitg.engine.train import TrainConfig, run_training

def main():
    ap = argparse.ArgumentParser(description="Train VITG on GEI dataset.")
    ap.add_argument("--train-dir", required=True, help="Path to training folder with class subdirs")
    ap.add_argument("--val-dir", required=True, help="Path to validation folder with class subdirs")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--max-epochs", type=int, default=50)
    args = ap.parse_args()

    cfg = TrainConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )
    run_training(cfg)

if __name__ == "__main__":
    main()
