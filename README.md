# 🧠 OpenCLIP Vision-Language Demo

A minimal demo using [OpenCLIP](https://github.com/mlfoundations/open_clip) to interpret a visual scene and suggest a robot action.

## 📦 Requirements

### Core Dependencies
```bash
# Essential packages
torch>=2.0.0
torchvision>=0.15.0
open-clip-torch>=2.0.0
Pillow>=10.0.0
```

### Video Processing
``` bash
# For nature_analyzer.py
opencv-python>=4.8.0
yt-dlp>=2023.11.16
pytube>=15.0.0  # (backup downloader)
```

### Utilities
``` bash
numpy>=1.24.0
requests>=2.25.0
```



## ⚙️ Setup
```bash
# Clone repository
git clone https://github.com/yourusername/vlm-demo.git
cd vlm-demo

# Create environment (optional but recommended)
conda create -n openclip python=3.10 -y
conda activate openclip

# Install dependencies
pip install -r requirements.txt
<<<<<<< HEAD
```

## Object Analysis
Run `python vlm_demo2.py` to see how the model interprets and describes an image.

**Example output:**
- a block lying flat on the table -> 91.82% ✅
- a tipped over block -> 3.88%
- a cube on its side -> 3.48%
- an upright block -> 0.69%
- a vertical standing block -> 0.13%


VLM top match: a block lying flat on the table  

Suggested action: pick up and stand the block upright.



## 🌄 Nature TimeLapse Analysis - Dynamic OpenCLIP Usage
Run `python nature_analyzer.py` to analyze nature videos and detect scenes.

### Features:
- Video Processing: Analyze local videos or download from YouTube
- Scene Detection: Identify nature scenes (milky way, storms, rainbows, landscapes)
- Smart Prompts: Includes confuser prompts to validate model intelligence
- Auto-Dependencies: Automatic package installation


### Usage: 
Analyze YouTube video
- python nature_analyzer.py --youtube "https://www.youtube.com/watch?v=dj44GB-FXbM"

Analyze local video  
- python nature_analyzer.py --video "nature_timelapse.mp4"

Faster processing
- python nature_analyzer.py --youtube "URL" --interval 60 --no-display


Example Output:
⏱️   12.5s | milky way galaxy night sky......... | 85.2%
⏱️   45.2s | storm clouds supercell............ | 92.1%
⏱️   78.9s | rainbow in sky.................... | 76.8%

📊 SCENE BREAKDOWN:
  storm clouds supercell................. 24 frames | avg: 88.3%
  milky way galaxy night sky............. 18 frames | avg: 82.1%
  rainbow in sky.........................  6 frames | avg: 74.5%
