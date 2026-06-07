# Incremenal memory build

This repository contains the PyTorch incremental open-set memory pipeline used for the thesis experiments.

# Dataset and checkpoint preparation

Adopted from [OW-DETR](https://github.com/akshitac8/OW-DETR).

`data/` and `checkpoints/` are not in the repo - prepare them locally before running notebooks or scripts

## Project setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

External codebases should live in `third_party/`

- `third_party/deformable_detr` — object detector
- `third_party/dinov3` — image encoder

If you clone them, place them under `third_party/` with the same folder names.

## Checkpoints

Download pretrained weights into `checkpoints/`.

### Deformable-DETR

- Code: [https://github.com/fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
- Checkpoint used by this project:

```
checkpoints/ddetr/r50_deformable_detr-checkpoint.pth
```

### DINOv3

- Weights: [https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
- Checkpoint used by this project:

```
checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## Dataset preparation

OW-DETR task splits (`t1_train.txt`, `test.txt`, etc.) should be in `data/OWDETR/VOC2007/ImageSets/`. Use the [OW-DETR](https://github.com/akshitac8/OW-DETR) repo to download them.

1. Create `JPEGImages` and `Annotations` directories:
  ```bash
   mkdir -p data/OWDETR/VOC2007/JPEGImages
   mkdir -p data/OWDETR/VOC2007/Annotations
  ```
2. Download COCO 2017 images and annotations from the [COCO dataset](https://cocodataset.org/#download).
3. Unzip `train2017`, `val2017`, and `annotations`. Layout under the repo root:
  ```
   ROOT_DIR/
   └── data/
       └── coco/
           ├── annotations/
           ├── train2017/
           └── val2017/
  ```
4. Move images into `JPEGImages`:
  ```bash
   cd ROOT_DIR/data
   mv coco/train2017/*.jpg OWDETR/VOC2007/JPEGImages/
   mv coco/val2017/*.jpg OWDETR/VOC2007/JPEGImages/
  ```
5. Copy COCO annotation JSONs into `Annotations/`:
  ```bash
   cp coco/annotations/instances_train2017.json OWDETR/VOC2007/Annotations/
   cp coco/annotations/instances_val2017.json OWDETR/VOC2007/Annotations/
  ```

Final layout:

```
ROOT_DIR/
└── data/
    └── OWDETR/
        └── VOC2007/
            ├── JPEGImages/
            ├── ImageSets/
            │   ├── t1_train.txt
            │   ├── test.txt
            │   └── ...
            └── Annotations/
                ├── instances_train2017.json
                └── instances_val2017.json
```

## Experiments

Run setups and parameter tables: `[notebooks/OWOD_EXPERIMENTS.md](notebooks/OWOD_EXPERIMENTS.md)`

Config sources:

- `scripts/experiment_runs_t1.py` — T1 memory / MLP / eval runs
- `scripts/experiment_runs_owod_cross.py` — OWOD tasks 1–4 cross-task runs

Batch runners (from repo root, venv active):

```bash
# T1 experiments
python3 scripts/batch_experiments_t1.py --stage P1_3_gates --skip-existing

# OWOD cross-task
python3 scripts/batch_experiments_owod_cross.py --skip-existing
```

Outputs `outputs/notebook/experiments/<run_dir>/`.