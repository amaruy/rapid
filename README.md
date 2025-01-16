# RAPID: Robust APT Detection and Investigation Using Context-Aware Deep Learning

A framework for detecting and investigating Advanced Persistent Threats (APTs) using context-aware deep learning. RAPID analyzes system-level events to detect and trace malicious activities in real-time.

## Features

- Context-aware anomaly detection
- Graph-based alert investigation
- Automated alert correlation and triage

## Requirements

### Hardware
- NVIDIA GPU with at least 8GB VRAM
- 32GB RAM recommended
- 100GB disk space for dataset and artifacts

### Software
- CUDA 11.x or higher
- Python 3.10
- See `requirements.txt` for Python dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rapid.git
cd rapid
```

2. Create and activate conda environment:
```bash
conda create -n rapid python=3.10
conda activate rapid
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The CADETS dataset (~40GB total decompressed) contains system-level provenance data with labeled APT activities.

### Automatic Download
```bash
make download_data
```

### Manual Download
```bash
cd data/cadets
mkdir -p raw_data

# Download dataset parts
curl "https://drive.google.com/uc?id=1AcWrYiBmgAqp7DizclKJYYJJBQbnDMfb" -o cadets_json_1.tar.gz
curl "https://drive.google.com/uc?id=1XLCEhf5DR8xw3S-Fimcj32IKnfzHFPJW" -o cadets_json_2.tar.gz
curl "https://drive.google.com/uc?id=1EycO23tEvZVnN3VxOHZ7gdbSCwqEZTI1" -o cadets_json_3.tar.gz

# Extract files
tar -xvf cadets_json_1.tar.gz
tar -xvf cadets_json_2.tar.gz
tar -xvf cadets_json_3.tar.gz

rm -f *.tar.gz
```

## Usage

### Complete Pipeline
Run the entire pipeline with:
```bash
make all
```

### Individual Stages
```bash
# Parse raw data
make parse_data

# Train models
make train_embedder  # Train graph embedder
make train_detector  # Train anomaly detector

# Run inference
make inference


