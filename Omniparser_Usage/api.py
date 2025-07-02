import io
import os
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image

# Ensure your local OmniParser folder is on PYTHONPATH
# (e.g. browser-use-agent/omniparser/OmniParser)
from OmniParser.util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _image_to_base64(img: Image.Image) -> str:
    if isinstance(img, str):
        return img
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def process_image(
    image_path: str,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = False,
    imgsz: int = 640,
) -> Dict[str, Any]:
    """
    Run OmniParser on an input image and return a JSON-serializable dict:
      - annotated_image: base64-encoded PNG
      - elements: List of {label, coords, caption, ...}
    """
    # 1) device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"OmniParser running on device: {device}")
    torch.cuda.empty_cache()

    # 2) load image
    img = Image.open(image_path)

    # 3) load models
    logger.info("Loading YOLO icon detector…")
    yolo = get_yolo_model(model_path=str(Path(__file__).parent / "../OmniParser/weights/icon_detect/model.pt"))

    logger.info("Loading Florence caption model…")
    captioner = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path=str(Path(__file__).parent / "../OmniParser/weights/icon_caption_florence"),
    )

    # 4) OCR
    logger.info("Running OCR…")
    (ocr_text, ocr_boxes), _ = check_ocr_box(
        img,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=use_paddleocr,
    )

    # 5) SOM labeling
    logger.info("Annotating screen elements…")
    labeled_img, label_coords, parsed_list = get_som_labeled_img(
        img,
        yolo,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_boxes,
        draw_bbox_config={
            "text_scale": 0.8 * (img.size[0] / 3200),
            "text_thickness": max(int(2 * (img.size[0] / 3200)), 1),
            "text_padding": max(int(3 * (img.size[0] / 3200)), 1),
            "thickness": max(int(3 * (img.size[0] / 3200)), 1),
        },
        caption_model_processor=captioner,
        ocr_text=ocr_text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )

    logger.info(f"Detected {len(parsed_list)} elements")
    if isinstance(parsed_list, dict):
        elements = parsed_list["elements"]
    elif isinstance(parsed_list, list):
        elements = parsed_list
    else:
        raise TypeError("parsed_list is neither dict nor list")

    result = [{"id": i, **e} for i, e in enumerate(elements)]

    return {
        #"annotated_image": _image_to_base64(labeled_img),
        "elements": result,
    }
