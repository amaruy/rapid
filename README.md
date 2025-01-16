# RAPID: Robust APT Detection and Investigation Using Context-Aware Deep Learning

A framework for detecting and investigating Advanced Persistent Threats (APTs) using context-aware deep learning. RAPID analyzes system-level events to detect and trace malicious activities.

## Requirements

- NVIDIA GPU with at least 8GB VRAM
- 32GB RAM recommended
- CUDA 12.4 or higher
- Python 3.10

## Quick Start

1. Clone and setup environment:
```bash
git clone https://github.com/yourusername/rapid.git
cd rapid
conda create -n rapid python=3.10
conda activate rapid
pip install -r requirements.txt
```

2. Download and prepare dataset:
```bash
# Download CADETS dataset (~40GB)
make download_data

# Parse and split dataset into train/test
make parse_data
```

3. Train model:
```bash
# Train both embedding and detection models
make train
```

4. Run inference and evaluate:
```bash
# Run detection pipeline and evaluate results
make inference
```

5. Evaluate results:
```bash
# Evaluate results
make evaluate
```


