# 🧠 OpenCLIP Vision-Language Demo

A minimal demo using [OpenCLIP](https://github.com/mlfoundations/open_clip) to interpret a visual scene and suggest a robot action.

## 📦 Requirements
- Python 3.10+
- PyTorch ≥ 2.5
- `open_clip_torch`
- `Pillow`

## ⚙️ Setup
```bash
conda create -n openclip python=3.10 -y
conda activate openclip
pip install torch torchvision open_clip_torch pillow
<<<<<<< HEAD
```

## Results
Run `python vlm_demo.py` to see how the model interprets and describes an image.

**Example output:**
- a block lying flat on the table -> 91.82% ✅
- a tipped over block -> 3.88%
- a cube on its side -> 3.48%
- an upright block -> 0.69%
- a vertical standing block -> 0.13%


VLM top match: a block lying flat on the table  

Suggested action: pick up and stand the block upright.