from dataclasses import dataclass, field

COCO_CLASSES: dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}

# OWOD split (category IDs)
TASK_SPLITS: dict[int, list[int]] = {
    1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    2: [22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    3: [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
    4: [67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],
}


@dataclass
class OWODSplit:

    task: int
    known_ids: list[int] = field(default_factory=list)
    unknown_ids: list[int] = field(default_factory=list)

    @property
    def known_names(self) -> list[str]:
        return [COCO_CLASSES[i] for i in self.known_ids]

    def is_known(self, category_id: int) -> bool:
        return category_id in self.known_ids


def get_split(task: int = 1) -> OWODSplit:
    if task < 1 or task > 4:
        raise ValueError(f"task must be 1-4, got {task}")

    known: list[int] = []
    for t in range(1, task + 1):
        known.extend(TASK_SPLITS[t])

    all_ids = set(COCO_CLASSES.keys())
    unknown = sorted(all_ids - set(known))

    return OWODSplit(task=task, known_ids=known, unknown_ids=unknown)


def resolve_split_task(cfg: dict) -> int | None:
    # OWOD split task (1–4) from split_task
    st = cfg.get("split_task")
    if st is None:
        return None

    st = int(st)
    if st < 1 or st > 4:
        raise ValueError(f"OWOD split_task must be 1–4, got {st}")
    return st
