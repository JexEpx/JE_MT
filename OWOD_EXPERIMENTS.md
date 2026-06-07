# Experiments

Source configs:

- `scripts/experiment_runs_t1.py`
- `scripts/experiment_runs_owod_cross.py`

Main experiments used in the project: prototype memory builds, exemplar memory builds, MHN/cosine evaluations, and OWOD cross-task runs.

## Experiment Counts


| Family            | Run IDs                      | Count | Purpose                                                     |
| ----------------- | ---------------------------- | ----- | ----------------------------------------------------------- |
| T1 MLP baseline   | `P0-01`                      | 1     | Parametric baseline on frozen DINO features (Task 1)        |
| Prototype memory  | `P1-01` to `P1-18`, `P1-all` | 19    | Prototype construction, open-set gate, online-update sweeps |
| Exemplar memory   | `E2-01` to `E2-09`           | 9     | Exemplar memory construction and update sweeps              |
| MHN / cosine eval | `S3-01` to `S3-10`           | 10    | Cosine vs MHN classification and MHN beta sweep             |
| MHN refine eval   | `S4-01` to `S4-04`           | 4     | MHN refinement on champion memory (eval-only)               |
| OWOD cross-task   | `A1-01` to `A4-03`           | 12    | Four-task OWOD comparison: MLP, prototype, exemplar         |
| Total listed here | -                            | 55    | Main experiments                                    |


Notes: OWOD cross-task runs import the P1-18 / E2-09 fit setups from `experiment_runs_t1.py` via `experiment_runs_owod_cross.py`, with per-task parameters in the OWOD table below.

## Shared Defaults


| Parameter                   | Value                                        |
| --------------------------- | -------------------------------------------- |
| Feature pipeline            | `norm`                                       |
| Features dir                | `outputs/notebook/encode_coco_gt_normalized` |
| T1 train split              | `data/OWDETR/VOC2007/ImageSets/t1_train.txt` |
| Test split                  | `data/OWDETR/VOC2007/ImageSets/test.txt`     |
| Device                      | `cuda`                                       |
| Random seed                 | `153`                                        |
| Open-set calibration method | `known_only`                                 |
| Top-1 threshold percentile  | `5.0`                                        |
| Margin threshold percentile | `5.0`                                        |
| Calibration max records     | `10000`                                      |
| Calibration seed            | `0`                                          |
| Default open-set gate       | `cosine_margin`                              |
| Default margin mode         | `class_aware`                                |


## T1 MLP Baseline (Experiment 0)


| Run ID  | Folder              | Train split    | Epochs | Batch | LR   | Hidden | Classifier  | Open-set gates            |
| ------- | ------------------- | -------------- | ------ | ----- | ---- | ------ | ----------- | ------------------------- |
| `P0-01` | `t1_mlp_classifier` | `t1_train.txt` | 15     | 512   | 1e-3 | 256    | MLP softmax | calibrated (`known_only`) |


Runner: `python3 scripts/batch_experiments_t1.py --run-id P0-01` or `--stage P0_mlp`.

## Prototype Memory Experiments


| Run ID   | Folder                                      | `n_support` | Init       | `k` | Online  | `online_max` | `tau_update` | `tau_new` | `alpha` | `online_min_cosine` | Gate variant   |
| -------- | ------------------------------------------- | ----------- | ---------- | --- | ------- | ------------ | ------------ | --------- | ------- | ------------------- | -------------- |
| `P1-01`  | `t1_proto_s24_k3_km_noOnline`               | 24          | `kmeans`   | 3   | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `P1-02`  | `t1_proto_s24_k3_ex_noOnline`               | 24          | `examples` | 3   | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `P1-03`  | `t1_proto_s32_k3_km_noOnline`               | 32          | `kmeans`   | 3   | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `P1-04`  | `t1_proto_s32_k3_ex_noOnline`               | 32          | `examples` | 3   | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `P1-05`  | `t1_proto_s48_k3_km_noOnline`               | 48          | `kmeans`   | 3   | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `P1-06`  | `t1_proto_s48_k3_ex_noOnline`               | 48          | `examples` | 3   | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `P1-07`  | `t1_proto_s32_k3_km_maha_noOnline`          | 32          | `kmeans`   | 3   | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `maha`         |
| `P1-08`  | `t1_proto_s32_k3_km_globalProto_noOnline`   | 32          | `kmeans`   | 3   | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `global_proto` |
| `P1-09`  | `t1_proto_s32_k3_km_on5k`                   | 32          | `kmeans`   | 3   | `true`  | 5000         | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `P1-10`  | `t1_proto_s32_k3_km_on10k`                  | 32          | `kmeans`   | 3   | `true`  | 10000        | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `P1-11`  | `t1_proto_s32_k3_km_on50k`                  | 32          | `kmeans`   | 3   | `true`  | 50000        | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `P1-12`  | `t1_proto_s32_k3_km_on5k_tu085`             | 32          | `kmeans`   | 3   | `true`  | 5000         | 0.85         | 0.45      | 0.05    | 0.20                | `class_aware`  |
| `P1-13`  | `t1_proto_s32_k3_km_on5k_tn040`             | 32          | `kmeans`   | 3   | `true`  | 5000         | 0.80         | 0.40      | 0.05    | 0.20                | `class_aware`  |
| `P1-14`  | `t1_proto_s32_k3_km_on5k_tu085_tn040`       | 32          | `kmeans`   | 3   | `true`  | 5000         | 0.85         | 0.40      | 0.05    | 0.20                | `class_aware`  |
| `P1-15`  | `t1_proto_s32_k3_km_on5k_a003`              | 32          | `kmeans`   | 3   | `true`  | 5000         | 0.80         | 0.45      | 0.03    | 0.20                | `class_aware`  |
| `P1-16`  | `t1_proto_s32_k3_km_on5k_omc25`             | 32          | `kmeans`   | 3   | `true`  | 5000         | 0.80         | 0.45      | 0.05    | 0.25                | `class_aware`  |
| `P1-17`  | `t1_proto_s32_k3_km_on5k_tu085_tn040_a003`  | 32          | `kmeans`   | 3   | `true`  | 5000         | 0.85         | 0.40      | 0.03    | 0.20                | `class_aware`  |
| `P1-18`  | `t1_proto_s32_k3_km_on5k_tu085_tn040_omc25` | 32          | `kmeans`   | 3   | `true`  | 5000         | 0.85         | 0.40      | 0.05    | 0.25                | `class_aware`  |
| `P1-all` | `t1_proto_s32_k3_km_onAll_ev5k`             | 32          | `kmeans`   | 3   | `true`  | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |


## T1 Batch Runner

Runner: `python3 scripts/batch_experiments_t1.py --output outputs/notebook/experiments/t1_results.csv`

Examples:

```bash
# Gate sweep (P1-09 … P1-18)
python3 scripts/batch_experiments_t1.py --stage P1_3_gates --skip-existing

# Single run
python3 scripts/batch_experiments_t1.py --run-id P1-18

# Re-eval only (memory must exist)
python3 scripts/batch_experiments_t1.py --run-id P1-18 --eval-only --force-eval

# T1 MLP baseline (P0-01)
python3 scripts/batch_experiments_t1.py --stage P0_mlp
```

Flags: `--all`, `--section N`, `--stage`, `--track proto|exem`, `--skip-existing`, `--eval-only`, `--fit-only`, `--collect-only`, `--force-eval`.

## Exemplar Memory Experiments


| Run ID  | Folder                                 | Exemplar cap | `n_support` | Online  | `online_max` | `tau_update` | `tau_new` | `alpha` | `online_min_cosine` | Gate variant   |
| ------- | -------------------------------------- | ------------ | ----------- | ------- | ------------ | ------------ | --------- | ------- | ------------------- | -------------- |
| `E2-01` | `t1_exem_cap20_noOnline`               | 20           | 5           | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `E2-02` | `t1_exem_cap50_noOnline`               | 50           | 5           | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `E2-03` | `t1_exem_cap100_noOnline`              | 100          | 5           | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `E2-04` | `t1_exem_cap50_maha_noOnline`          | 50           | 5           | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `maha`         |
| `E2-05` | `t1_exem_cap50_globalProto_noOnline`   | 50           | 5           | `false` | 0            | 0.70         | 0.50      | 0.10    | 0.15                | `global_proto` |
| `E2-06` | `t1_exem_cap50_on5k`                   | 50           | 5           | `true`  | 5000         | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `E2-07` | `t1_exem_cap50_on10k`                  | 50           | 5           | `true`  | 10000        | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `E2-08` | `t1_exem_cap50_on50k`                  | 50           | 5           | `true`  | 50000        | 0.70         | 0.50      | 0.10    | 0.15                | `class_aware`  |
| `E2-09` | `t1_exem_cap50_on5k_tu085_tn040_omc25` | 50           | 5           | `true`  | 5000         | 0.85         | 0.40      | 0.05    | 0.25                | `class_aware`  |


## MHN / Cosine Evaluation Experiments


| Run ID  | Folder                | Memory source | Classifier | `mhn_beta` | `use_mhn_refine` | Gate class source |
| ------- | --------------------- | ------------- | ---------- | ---------- | ---------------- | ----------------- |
| `S3-01` | `t1_s3_proto_cosine`  | prototype     | cosine     | 20.0       | `false`          | cosine            |
| `S3-02` | `t1_s3_proto_mhn`     | prototype     | MHN        | 30.0       | `false`          | MHN               |
| `S3-03` | `t1_s3_exem_cosine`   | exemplar      | cosine     | 20.0       | `false`          | cosine            |
| `S3-04` | `t1_s3_exem_mhn`      | exemplar      | MHN        | 30.0       | `false`          | MHN               |
| `S3-05` | `t1_s3_proto_mhn_b05` | prototype     | MHN        | 5.0        | `false`          | MHN               |
| `S3-06` | `t1_s3_proto_mhn_b10` | prototype     | MHN        | 10.0       | `false`          | MHN               |
| `S3-07` | `t1_s3_proto_mhn_b20` | prototype     | MHN        | 20.0       | `false`          | MHN               |
| `S3-08` | `t1_s3_exem_mhn_b05`  | exemplar      | MHN        | 5.0        | `false`          | MHN               |
| `S3-09` | `t1_s3_exem_mhn_b10`  | exemplar      | MHN        | 10.0       | `false`          | MHN               |
| `S3-10` | `t1_s3_exem_mhn_b20`  | exemplar      | MHN        | 20.0       | `false`          | MHN               |


Note: These are eval-only runs on fixed prototype or exemplar memories.

## MHN Refine Evaluation Experiments


| Run ID  | Folder                      | Memory source     | Classifier | `mhn_beta` | `use_mhn_refine` | Gate class source |
| ------- | --------------------------- | ----------------- | ---------- | ---------- | ---------------- | ----------------- |
| `S4-01` | `t1_s4_proto_cosine_refine` | P1-18 (prototype) | cosine     | 20.0       | `true`           | cosine            |
| `S4-02` | `t1_s4_proto_mhn_refine`    | P1-18 (prototype) | MHN        | 20.0       | `true`           | MHN               |
| `S4-03` | `t1_s4_exem_cosine_refine`  | E2-09 (exemplar)  | cosine     | 20.0       | `true`           | cosine            |
| `S4-04` | `t1_s4_exem_mhn_refine`     | E2-09 (exemplar)  | MHN        | 20.0       | `true`           | MHN               |


Runner: `python3 scripts/batch_experiments_t1.py --stage S4_refine_proto` or `--stage S4_refine_exem`.

## OWOD Cross-Task Experiments


| Run ID  | Task | Track     | Folder                                      | Construction / training | Train split                    | Resume | `online_max` | Calibration split              | Eval classifier | Eval gate            |
| ------- | ---- | --------- | ------------------------------------------- | ----------------------- | ------------------------------ | ------ | ------------ | ------------------------------ | --------------- | -------------------- |
| `A1-01` | 1    | MLP       | `t1_mlp_cumulative`                         | MLP from scratch        | `t1_owod_train_cumulative.txt` | no     | n/a          | `t1_owod_train_cumulative.txt` | MLP             | calibrated open-set  |
| `A1-02` | 1    | prototype | `t1_proto_s32_k3_km_on5k_tu085_tn040_omc25` | P1-18 prototype memory  | `t1_train`                     | no     | 5000         | `t1_train.txt`                 | MHN beta 20.0   | MHN-aligned open-set |
| `A1-03` | 1    | exemplar  | `t1_exem_cap50_on5k_tu085_tn040_omc25`      | E2-09 exemplar memory   | `t1_train`                     | no     | 5000         | `t1_train.txt`                 | MHN beta 20.0   | MHN-aligned open-set |
| `A2-01` | 2    | MLP       | `t2_mlp_cumulative`                         | MLP from scratch        | `t2_owod_train_cumulative.txt` | no     | n/a          | `t2_owod_train_cumulative.txt` | MLP             | calibrated open-set  |
| `A2-02` | 2    | prototype | `t2_proto_s32_k3_km_on5k_tu085_tn040_omc25` | P1-18 prototype memory  | `t2_train`                     | yes    | 10000        | `t2_owod_train_cumulative.txt` | MHN beta 20.0   | MHN-aligned open-set |
| `A2-03` | 2    | exemplar  | `t2_exem_cap50_on5k_tu085_tn040_omc25`      | E2-09 exemplar memory   | `t2_train`                     | yes    | 10000        | `t2_owod_train_cumulative.txt` | MHN beta 20.0   | MHN-aligned open-set |
| `A3-01` | 3    | MLP       | `t3_mlp_cumulative`                         | MLP from scratch        | `t3_owod_train_cumulative.txt` | no     | n/a          | `t3_owod_train_cumulative.txt` | MLP             | calibrated open-set  |
| `A3-02` | 3    | prototype | `t3_proto_s32_k3_km_on5k_tu085_tn040_omc25` | P1-18 prototype memory  | `t3_train`                     | yes    | 15000        | `t3_owod_train_cumulative.txt` | MHN beta 20.0   | MHN-aligned open-set |
| `A3-03` | 3    | exemplar  | `t3_exem_cap50_on5k_tu085_tn040_omc25`      | E2-09 exemplar memory   | `t3_train`                     | yes    | 15000        | `t3_owod_train_cumulative.txt` | MHN beta 20.0   | MHN-aligned open-set |
| `A4-01` | 4    | MLP       | `t4_mlp_cumulative`                         | MLP from scratch        | `t4_owod_train_cumulative.txt` | no     | n/a          | `t4_owod_train_cumulative.txt` | MLP             | calibrated open-set  |
| `A4-02` | 4    | prototype | `t4_proto_s32_k3_km_on5k_tu085_tn040_omc25` | P1-18 prototype memory  | `t4_train`                     | yes    | 20000        | `t4_owod_train_cumulative.txt` | MHN beta 20.0   | closed-set           |
| `A4-03` | 4    | exemplar  | `t4_exem_cap50_on5k_tu085_tn040_omc25`      | E2-09 exemplar memory   | `t4_train`                     | yes    | 20000        | `t4_owod_train_cumulative.txt` | MHN beta 20.0   | closed-set           |


Runner: `python3 scripts/batch_experiments_owod_cross.py --output outputs/notebook/experiments/A_owod.csv`

Note: MLP train and calibration both use `t{k}_owod_train_cumulative.txt` (union of `t1_train` … `t{k}_train`), generated under `outputs/notebook/experiments/owod_cross_task_incremental/splits/`. Memory tracks use per-task `t{k}_train.txt` for fit and cumulative splits for gate calibration on tasks 2–4.

## OWOD Memory / MHN Settings


| Parameter                          | Prototype OWOD      | Exemplar OWOD       |
| ---------------------------------- | ------------------- | ------------------- |
| Source recipe                      | `P1-18`             | `E2-09`             |
| `use_mhn_classify`                 | `true`              | `true`              |
| `use_mhn_refine`                   | `false`             | `false`             |
| `mhn_beta`                         | `20.0`              | `20.0`              |
| `open_set_gate_use_cosine_class`   | `false`             | `false`             |
| Gate class source                  | MHN-predicted class | MHN-predicted class |
| `use_global_threshold` T1-T3       | `true`              | `true`              |
| `use_proto_margin` T1-T3           | `true`              | `true`              |
| `use_global_threshold` T4 MHN eval | `false`             | `false`             |
| `use_proto_margin` T4 MHN eval     | `false`             | `false`             |


