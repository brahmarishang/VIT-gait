import cv2, os
from pathlib import Path

def build_silhouettes(raw_dir, out_dir, history=500, var_threshold=16, remove_shadows=True):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for subj in sorted(d for d in raw_dir.iterdir() if d.is_dir()):
        for view in sorted(d for d in subj.iterdir() if d.is_dir()):
            # One subtractor per sequence keeps local background stable
            mog2 = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                      detectShadows=remove_shadows)
            out_seq = out_dir / subj.name / view.name
            out_seq.mkdir(parents=True, exist_ok=True)

            frames = sorted(view.glob("*.avi")) or sorted(view.glob("*.mp4")) or sorted(view.glob("*.png"))
            if not frames:
                # handle video container or image frames
                cap = cv2.VideoCapture(str(view / "video.avi"))
                idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fg = mog2.apply(gray)
                    if remove_shadows:
                        fg[fg == 127] = 0
                    # binarise + morph refine (quick pass)
                    _, binm = cv2.threshold(fg, 0, 255, cv2.THRESH_OTSU)
                    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, (3,3))
                    cv2.imwrite(str(out_seq / f"{idx:05d}.png"), binm)
                    idx += 1
                cap.release()
            else:
                # if frames already extracted, iterate images
                for idx, fp in enumerate(frames):
                    img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
                    fg = mog2.apply(img)
                    if remove_shadows:
                        fg[fg == 127] = 0
                    _, binm = cv2.threshold(fg, 0, 255, cv2.THRESH_OTSU)
                    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, (3,3))
                    cv2.imwrite(str(out_seq / f"{idx:05d}.png"), binm)

if __name__ == "__main__":
    build_silhouettes("data/casia-b/raw", "data/casia-b/silhouettes")
