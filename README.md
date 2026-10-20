# LGGA

This is the official code repository for LGGA, our language-guided framework for dynamic hand gesture authentication.

The project retains the original `conf/`, `sh/`, and `src/` layout and the `src/main/train.py` training entry. The configuration in `conf/confS/dino/Test_dinov3_clip_mask_guided_physio_behavior_modular_direct384_temporalmotion_windowpool_histproto.yaml` covers BR training and BR/Real-DHGA verification. Datasets and pretrained weights are not included.

## Preparation

```bash
conda create -n mydhga python=3.9
conda activate mydhga

# Example for CUDA 12.1; select the PyTorch build matching your server.
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The `timm` installation must provide `vit_small_plus_patch16_dinov3.lvd1689m`; `open_clip_torch` must be able to load `ViT-B-16` with the `openai` weights. Make both pretrained weights available in the library caches before offline runs. To reproduce the reported setup as closely as possible, use the same library versions and hardware environment as the original training run.

The config uses `/data/work/datasets/DHGAdataset/DHG-Auth/` for BR training/testing and `/data/work/datasets/DHGAdataset/Real-DHGA/` for Real testing. If necessary, update only `dataset.train_sources`, `dataset.eval_sources`, and `dataset.keypoint_root` in the YAML. BR training requires the RGB frames, hand-mask ground truth, keypoints, and `train_MG.csv`. Evaluation requires the six BR cross-session pair CSVs and ten Real-DHGA single-session pair CSVs listed in the config. The model predicts the mask at test time; ground-truth masks supervise training only.

## Train

From the project root:

```bash
bash sh/shS/train_mask_guided_physio_behavior_modular_direct384_temporalmotion_windowpool_histproto.sh 0
```

The argument is the GPU ID. The entry still calls `src/main/train.py`. An equivalent direct invocation from `sh/` is:

```bash
cd sh
python ../src/main/train.py \
    --mode train \
    --gpuid 0 \
    --conf_file confS/dino/Test_dinov3_clip_mask_guided_physio_behavior_modular_direct384_temporalmotion_windowpool_histproto.yaml
```

The config samples 22 frames per video. Training evaluates the configured BR and Real-DHGA pair sets according to `evaluate.eval_interval`, saves one `*_best.pth` for each test subset, and writes `last.pth` under `workdir/checkpoints/<task>_train/`. Scalar training metrics are logged under `workdir/log/` and `workdir/tb/`.

## Test / Evaluate

Training already runs evaluation. To test a saved checkpoint separately, run from the project root:

```bash
bash sh/shS/test_mask_guided_physio_behavior_modular_direct384_temporalmotion_windowpool_histproto.sh 0 \
    workdir/checkpoints/mask_guided02P02_physio_behavior612h02_modular_direct384_temporalmotion_windowpool_histproto_22_050502_brvis7_train/last.pth
```

The first argument is the GPU ID; the second is a checkpoint file or a directory containing `*_best.pth` files. Alternatively, set `model.resume_from` in the YAML and invoke `src/main/train.py --mode test` using the same `--gpuid` and `--conf_file` arguments. The standalone tester reports per-subset and average EER and saves its results to `workdir/test_results/<task>/checkpoint_results.csv`.

Standalone testing reads `train_MG.csv` to recover the training identity-class count needed to initialize the classification head; it does not load training videos.

**Under review.** This work is currently under peer review. The complete model implementation will be released after the review process. Until then, the training and testing commands above are provided for reference and require the unreleased model file.
