
import argparse
from vitg.engine.eval import evaluate_checkpoint

def main():
    ap = argparse.ArgumentParser(description="Evaluate a VITG checkpoint on a GEI folder.")
    ap.add_argument("--ckpt", required=True, help="Path to vitg_best.pt")
    ap.add_argument("--data-dir", required=True, help="Folder with class subdirs of GEIs")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    evaluate_checkpoint(
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()
