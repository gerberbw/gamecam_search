#!/usr/bin/env python3
"""Search a directory of images with YOLO and print images containing configured labels.

Configuration via environment variables (use a .env file):
- YOLO_MODEL: model name or path (default: yolov8n.pt)
- YOLO_LABELS: comma-separated labels to match (required)
- CONFIDENCE_THRESHOLD: float 0-1 (default: 0.25)
- RECURSIVE: true/false whether to search subdirectories (default: true)
"""
import argparse
import os
from pathlib import Path
import sys
from dotenv import load_dotenv


def is_image_file(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_config():
    load_dotenv()
    cfg = {
        "model": os.getenv("YOLO_MODEL", "yolov8n.pt"),
        "labels": [s.strip() for s in os.getenv("YOLO_LABELS", "").split(",") if s.strip()],
        "conf_thresh": float(os.getenv("CONFIDENCE_THRESHOLD", "0.25")),
        "recursive": os.getenv("RECURSIVE", "true").lower() in ("1", "true", "yes"),
    }
    return cfg


def find_images(root: Path, recursive: bool):
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and is_image_file(p):
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and is_image_file(p):
                yield p


def main():
    parser = argparse.ArgumentParser(description="Search directory images with YOLO for configured labels")
    parser.add_argument("path", help="Directory path to search")
    args = parser.parse_args()

    cfg = load_config()
    if not cfg["labels"]:
        print("YOLO_LABELS not configured. Set YOLO_LABELS in .env (comma-separated).", file=sys.stderr)
        sys.exit(2)

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Failed to import ultralytics. Install dependencies: pip install -r requirements.txt", file=sys.stderr)
        raise

    model = YOLO(cfg["model"])

    # model.names may be dict or list
    names = getattr(model, "names", None) or {}

    root = Path(args.path)
    if not root.exists() or not root.is_dir():
        print(f"Path not found or not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    target_labels = {lbl.lower() for lbl in cfg["labels"]}

    for img_path in find_images(root, cfg["recursive"]):
        try:
            results = model(str(img_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}", file=sys.stderr)
            continue

        if not results:
            continue

        r = results[0]
        found = set()

        # Boxes may be empty; try to read class indices
        boxes = getattr(r, "boxes", None)
        if boxes is not None and hasattr(boxes, "cls"):
            try:
                cls_tensor = boxes.cls
                # convert to ints
                cls_list = [int(x) for x in cls_tensor.cpu().numpy().reshape(-1)]
            except Exception:
                cls_list = []

            for cid in cls_list:
                label = names.get(cid, str(cid)) if isinstance(names, dict) else (names[cid] if cid < len(names) else str(cid))
                if label.lower() in target_labels:
                    found.add(label)

        if found:
            print(f"{img_path}: {', '.join(sorted(found))}")


if __name__ == "__main__":
    main()
