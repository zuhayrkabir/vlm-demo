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
