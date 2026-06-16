# inventscan

Extract item inventories from photos. Takes images of rooms or damaged property, runs object detection, and outputs a structured report.

## What it does

Point it at a photo of a room (or a folder of photos). It detects household items (furniture, electronics, appliances, personal items) using a pre-trained YOLOv8 model, groups duplicates, estimates condition, and generates a JSON inventory report.

This is a small tool, not an insurance claims platform. It works best on well-lit photos with visible household objects. The condition estimate is a heuristic based on detection confidence, not actual damage assessment.

## Usage

```bash
pip install -r requirements.txt

# Single photo
python3 inventscan.py photo.jpg

# Multiple photos or a folder
python3 inventscan.py room1.jpg room2.jpg --output report.json
python3 inventscan.py ./damage_photos/ --output claim_report.json
```

On first run it downloads a ~6MB YOLOv8 ONNX model and COCO labels (~5 seconds on most connections). Subsequent runs are offline.

## Output

JSON report with item-by-item breakdown, quantities, conditions, and summary stats:

```json
{
  "total_items": 14,
  "unique_item_types": 8,
  "inventory": [
    {"item": "chair", "quantity": 4, "condition": "good"},
    {"item": "dining table", "quantity": 1, "condition": "good"},
    {"item": "tv", "quantity": 1, "condition": "fair"}
  ]
}
```

## Caveats

- Uses COCO pre-trained weights, not fine-tuned on damage data. Will miss items not in COCO (e.g., specific appliances, custom fixtures).
- Condition rating is heuristic (based on detection confidence). A damaged table that is still clearly a table will score "good" on detection.
- This is a proof-of-concept. For production use, fine-tune on domain-specific damage photos.
- One script, ~200 lines of inference logic plus CLI wrapper. Not production-grade error handling.
