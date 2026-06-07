import numpy as np
from PIL import Image

def clip_boxes_xyxy(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes.astype(np.float32)

    boxes = boxes.astype(np.float32)
    # Use image bounds [0, w] / [0, h] for xyxy where x1/y1 are max edges
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


def valid_box_mask(boxes: np.ndarray, min_side: float = 2.0) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=bool)

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return (w >= min_side) & (h >= min_side)


def crop_image_by_box(image: np.ndarray, box) -> Image.Image:
    x0, y0, x1, y1 = map(int, box)
    return Image.fromarray(image).crop((x0, y0, x1, y1))

def crops_from_boxes(image: np.ndarray, boxes_xyxy) -> list[Image.Image]:
    return [crop_image_by_box(image, box) for box in boxes_xyxy]