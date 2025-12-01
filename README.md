# VITG 


## Structure
```
vitg_repo/
├─ src/
│  
│  ├─ __init__.py
│  ├─ data/
│  │  └─ datasets.py
│  ├─ models/
│  │  └─ vitg.py
│  └─ engine/
│  │    ├─ train.py
│  │    └─ eval.py
│  └─ scripts/
│   │    ├─ train.py
│  │    └─ eval.py
│  └─ tool/
│     ├─ build_silhouettes.py
│     └─ GEI.py
├─ requirements.txt
├─ pyproject.toml
└─ setup.cfg
```
## 📚 Dataset & Preparation
CASIA-B (124 subjects, 11 viewpoints, NM/CL/BG) download from the official provider http://english.ia.cas.cn/db/201610/t20161026_169403.html.
We provide simple tools to build silhouettes (MOG2) and GEIs:
```bash
python  src/tool/build_silhouettes.py
python  src/tool/GEI.py
```


## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # runtime deps
python  src/tool/build_silhouettes.py
python  src/tool/GEI.py



# optional: pip install black isort flake8 ruff

# Train
python src/scripts/train.py --train-dir /path/to/train --val-dir /path/to/val --img-size 128 --batch-size 16

# Evaluate
python src/scripts/eval.py --ckpt artifacts/vitg_best.pt --data-dir /path/to/test --img-size 128
```


