# inventscan
> Extract item inventories from photos using YOLOv8 ONNX object detection

```
$ inventscan photo.jpg
[detect] chair x 3 (conf 0.92, 0.88, 0.85)
[detect] laptop x 1 (conf 0.94)
[detect] monitor x 2 (conf 0.91, 0.87)
[detect] mug x 4 (conf 0.79-0.93)
Total items: 10
```

## Overview
A CLI tool that runs YOLOv8 object detection on photos and outputs a structured inventory (item names + counts + confidence scores). Uses ONNX Runtime for fast CPU/GPU inference with no PyTorch dependency at runtime.

## Features
- YOLOv8 object detection via ONNX Runtime
- Outputs structured JSON inventory (item, count, confidence)
- CLI + Python API
- Pre-trained COCO weights, swap in custom models

## Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white) ![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Installation
```bash
git clone https://github.com/MeGaurav4/inventscan.git
cd inventscan
pip install -r requirements.txt
# Download yolov8n.onnx into models/
```

## Usage
```bash
inventscan photo.jpg
inventscan photo.jpg --format json --threshold 0.5
```

## License
MIT