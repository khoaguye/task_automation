#!/usr/bin/env python3
import sys
import json
import argparse
import warnings
import logging
import torch

from pathlib import Path
from PIL import Image

# silence warnings
warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

# import your API
from Omniparser_Usage.api import process_image

def parse_args():
    p = argparse.ArgumentParser(description="OmniParser Runner")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Path for output JSON")
    p.add_argument("--box_threshold", type=float, default=0.05)
    p.add_argument("--iou_threshold", type=float, default=0.1)
    p.add_argument("--use_paddleocr", action="store_true")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()

def main():
    args = parse_args()
    try:
        result = process_image(
            args.input,
            box_threshold=args.box_threshold,
            iou_threshold=args.iou_threshold,
            use_paddleocr=args.use_paddleocr,
            imgsz=args.imgsz,
        )
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"Results saved to {args.output}")
    except Exception as e:
        logging.error(f"Failed: {e}")
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
