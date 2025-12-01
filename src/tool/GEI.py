import numpy as np, cv2
from pathlib import Path

def build_geis(sil_dir, gei_dir, size=(128,128)):
    sil_dir, gei_dir = Path(sil_dir), Path(gei_dir)
    for seq in sorted(sil_dir.rglob("*")):
        if not seq.is_dir() or len(list(seq.glob("*.png"))) == 0:
            continue
        imgs = []
        for fp in sorted(seq.glob("*.png")):
            im = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)
            im = (im > 0).astype(np.float32)
            imgs.append(im)
        if not imgs:
            continue
        # naive cycle assumption (use your cycle detector if available)
        gei = np.mean(np.stack(imgs, axis=0), axis=0)
        out = gei_dir / seq.relative_to(sil_dir)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "gei.png"), (gei * 255).astype(np.uint8))

if __name__ == "__main__":
    build_geis("data/casia-b/silhouettes", "data/casia-b/gei", size=(128,128))
