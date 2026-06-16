#!/usr/bin/env python3
"""
inventscan: extract item inventories from photos.
Takes images of rooms or damaged property, detects household items
using a pre-trained object detection model, and outputs a structured
inventory report suitable for insurance claims processing.

Usage:
    python3 inventscan.py photo.jpg
    python3 inventscan.py photo1.jpg photo2.jpg --output report.json
    python3 inventscan.py folder/
"""

import json
import sys
import io
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Model download & inference
# ---------------------------------------------------------------------------

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
MODEL_CACHE = Path.home() / ".inventscan" / "yolov8n.onnx"
COCO_LABELS_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
COCO_CACHE = Path.home() / ".inventscan" / "coco_labels.txt"

# Household-relevant COCO categories (mapped to inventory-friendly names)
HOUSEHOLD_CATEGORIES = {
    0: "person", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie",
    28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url.split('/')[-1]}...", file=sys.stderr)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        downloaded = 0
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)


def _load_labels():
    if not COCO_CACHE.exists():
        _download(COCO_LABELS_URL, COCO_CACHE)
    return [l.strip() for l in COCO_CACHE.read_text().splitlines() if l.strip()]


def _load_model():
    if not MODEL_CACHE.exists():
        _download(MODEL_URL, MODEL_CACHE)
    import onnxruntime as ort
    session = ort.InferenceSession(str(MODEL_CACHE))
    input_name = session.get_inputs()[0].name
    return session, input_name


def _preprocess(img, target_size=640):
    orig = img.size
    img = img.convert("RGB")
    img_resized = img.resize((target_size, target_size))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, H, W)
    return arr, orig


def _postprocess(output, orig_size, conf_thresh=0.4, iou_thresh=0.45):
    """Parse ONNX YOLOv8 output into detections."""
    output = np.squeeze(output[0], axis=0)  # (84, 8400)
    orig_w, orig_h = orig_size
    scale_x = orig_w / 640
    scale_y = orig_h / 640

    boxes, scores, class_ids = [], [], []
    for i in range(output.shape[1]):
        scores_i = output[4:, i]
        class_id = int(np.argmax(scores_i))
        score = float(scores_i[class_id])
        if score < conf_thresh:
            continue

        cx, cy, w, h = output[:4, i]
        x1 = max(0, (cx - w / 2) * scale_x)
        y1 = max(0, (cy - h / 2) * scale_y)
        x2 = min(orig_w, (cx + w / 2) * scale_x)
        y2 = min(orig_h, (cy + h / 2) * scale_y)

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    if not boxes:
        return []

    # NMS
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    indices = _nms(boxes_np, scores_np, iou_thresh)

    detections = []
    for idx in indices:
        detections.append({
            "class_id": class_ids[idx],
            "label": COCO_LABELS[class_ids[idx]] if class_ids[idx] < len(COCO_LABELS) else f"class_{class_ids[idx]}",
            "confidence": round(float(scores_np[idx]), 3),
            "bbox": [round(float(x), 1) for x in boxes_np[idx].tolist()],
            "category": HOUSEHOLD_CATEGORIES.get(class_ids[idx], "other"),
        })
    return detections


def _nms(boxes, scores, iou_thresh):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[1:][iou <= iou_thresh]
    return keep


def _estimate_condition(detections, img):
    """Heuristic: estimate item condition based on detection confidence."""
    for d in detections:
        confidence = d["confidence"]
        if confidence < 0.5:
            d["condition"] = "poor"
        elif confidence < 0.7:
            d["condition"] = "fair"
        else:
            d["condition"] = "good"
    return detections


def _assign_quantity(detections):
    """Group same-class detections and assign quantities."""
    grouped = {}
    for d in detections:
        cat = d["category"]
        if cat == "person":
            continue
        label = d["label"]
        if label not in grouped:
            grouped[label] = {"count": 0, "detections": []}
        grouped[label]["count"] += 1
        grouped[label]["detections"].append(d)

    inventory = []
    for label, data in sorted(grouped.items()):
        conditions = [d["condition"] for d in data["detections"]]
        # majority condition
        condition = max(set(conditions), key=conditions.count)
        inventory.append({
            "item": label,
            "quantity": data["count"],
            "condition": condition,
            "note": f"Detected {data['count']} instance(s) in photo",
        })
    return inventory


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(inventory, source, output_path=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    item_count = sum(i["quantity"] for i in inventory)
    unique_items = len(inventory)

    report = {
        "report_type": "photo_inventory",
        "generated": timestamp,
        "source": source,
        "summary": {
            "total_items": item_count,
            "unique_item_types": unique_items,
        },
        "inventory": inventory,
    }

    # Print text summary
    print(f"\n  Source: {source}")
    print(f"  Items detected: {item_count} ({unique_items} unique types)\n")
    print(f"  {'Item':<24} {'Qty':<6} {'Condition':<12}")
    print(f"  {'-'*24} {'-'*6} {'-'*12}")
    for item in inventory:
        print(f"  {item['item']:<24} {item['quantity']:<6} {item['condition']:<12}")
    print()

    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2))
        print(f"  Report saved: {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print("Usage:")
        print("  python3 inventscan.py <photo_or_folder> [--output report.json]")
        print("\nExamples:")
        print("  python3 inventscan.py damage_photo.jpg")
        print("  python3 inventscan.py ./room_photos/ --output claim_001.json")
        return

    # Parse --output
    output_path = None
    sources = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            skip_next = True
        else:
            sources.append(a)

    # Resolve image paths
    image_paths = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            image_paths.extend(sorted(p.glob("*.jpg")) + sorted(p.glob("*.jpeg")) + sorted(p.glob("*.png")) + sorted(p.glob("*.webp")))
        elif p.is_file():
            image_paths.append(p)

    if not image_paths:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    print(f"  inventscan: photo inventory extractor")
    print(f"  Images: {len(image_paths)}")

    # Load model (first run downloads ~6MB ONNX model + labels)
    global COCO_LABELS
    print(f"  Loading model...", file=sys.stderr)
    COCO_LABELS = _load_labels()
    session, input_name = _load_model()

    all_inventory = []
    total_items = 0

    for img_path in image_paths:
        print(f"\n  Processing: {img_path.name}", file=sys.stderr)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skip {img_path.name}: {e}", file=sys.stderr)
            continue

        tensor, orig_size = _preprocess(img)
        output = session.run(None, {input_name: tensor})
        detections = _postprocess(output, orig_size)
        detections = _estimate_condition(detections, img)
        inventory = _assign_quantity(detections)

        if inventory:
            all_inventory.extend(inventory)
            total_items += sum(i["quantity"] for i in inventory)
        else:
            print(f"  No items detected in {img_path.name}", file=sys.stderr)

    if all_inventory:
        # Merge duplicates across photos
        merged = {}
        for item in all_inventory:
            k = item["item"]
            if k in merged:
                merged[k]["quantity"] += item["quantity"]
            else:
                merged[k] = dict(item)
        report_inventory = sorted(merged.values(), key=lambda x: -x["quantity"])

        report = {
            "report_type": "photo_inventory",
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": ", ".join(str(p) for p in image_paths[:5])
                       + (f" and {len(image_paths)-5} more" if len(image_paths) > 5 else ""),
            "summary": {
                "total_items": total_items,
                "unique_item_types": len(report_inventory),
                "photos_processed": len(image_paths),
            },
            "inventory": report_inventory,
        }

        print(f"\n{'='*60}")
        print(f"  INVENTORY REPORT")
        print(f"{'='*60}")
        print(f"  Total items: {total_items}")
        print(f"  Unique types: {len(report_inventory)}")
        print(f"\n  {'Item':<24} {'Qty':<6} {'Condition':<12}")
        print(f"  {'-'*24} {'-'*6} {'-'*12}")
        for item in report_inventory:
            print(f"  {item['item']:<24} {item['quantity']:<6} {item['condition']:<12}")

        if output_path:
            Path(output_path).write_text(json.dumps(report, indent=2))
            print(f"\n  Report saved: {output_path}")
    else:
        print("\n  No items detected across any photos.")
        print("  Try photos with more visible household objects (furniture, electronics, etc.)")


if __name__ == "__main__":
    main()
