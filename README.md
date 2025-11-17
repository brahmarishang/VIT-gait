# VITG 

This folder contains a GitHub-friendly export of the notebook **mtech.ipynb**.

## Structure
```
vitg_repo/
├─ src/
├─ requirements.txt       # packages inferred from imports
└─ README.md
```

## Usage
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/vitg_main.py
```

> Notes:
> - Markdown cells are preserved as comments in `vitg_main.py`.
> - Jupyter magics were removed. Adjust paths and entry points as needed.
> - You can split `src/vitg_main.py` into modules (e.g., `models/`, `data/`, `train.py`) later.


## Modular layout
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
│     ├─ train.py
│     └─ eval.py
├─ requirements.txt
├─ pyproject.toml
└─ setup.cfg
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # runtime deps
# optional: pip install black isort flake8 ruff

# Train
python src/scripts/train.py --train-dir /path/to/train --val-dir /path/to/val --img-size 128 --batch-size 16

# Evaluate
python src/scripts/eval.py --ckpt artifacts/vitg_best.pt --data-dir /path/to/test --img-size 128
```

## Notes
