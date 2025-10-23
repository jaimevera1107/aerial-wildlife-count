#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_deformable_detr.py
MMDetection - Deformable DETR con backbone Swin-T (por defecto) o ResNeXt.

Requisitos:
 - MMDetection instalado
 - pip install pyyaml
"""

import argparse
from pathlib import Path
import yaml
from mmengine.config import Config
from mmengine.runner import Runner

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_train_final(aug_cfg):
    out = Path(aug_cfg.get("output_dir", "data/outputs")).resolve()
    d = out / "mirror_clean" / "train_final"
    j = d / "train_final.json"
    if not d.exists() or not j.exists():
        raise FileNotFoundError(f"No encontré {d} o {j}. Corre antes el pipeline de augment.")
    return d, j

def find_val_test(qual_cfg):
    splits = qual_cfg.get("splits", {})
    val_json = Path(splits["val"]).resolve()
    test_json = Path(splits["test"]).resolve()
    if not val_json.exists() or not test_json.exists():
        raise FileNotFoundError("No encontré val/test JSON en quality_config.yaml.")
    return val_json, test_json

def build_cfg(backbone: str, imgsz: int, epochs: int,
              train_img_dir: Path, train_json: Path,
              val_json: Path, test_json: Path,
              workdir: Path) -> Config:

    if backbone.lower().startswith("swin"):
        base_model = "mmdet::_base_/models/deformable_detr_swin-t.py"
    else:
        base_model = "mmdet::_base_/models/deformable_detr_r50.py"  # o resnext si tienes config

    cfg = Config(dict(
        _base_=[
            base_model,
            "mmdet::_base_/datasets/coco_instance.py",
            "mmdet::_base_/schedules/schedule_1x.py",
            "mmdet::_base_/default_runtime.py",
        ],
        default_scope="mmdet",

        dataset_type="CocoDataset",

        train_dataloader=dict(
            batch_size=4,
            num_workers=8,
            persistent_workers=True,
            sampler=dict(type="DefaultSampler", shuffle=True),
            dataset=dict(
                type="CocoDataset",
                ann_file=str(train_json),
                data_prefix=dict(img=str(train_img_dir)),
                filter_cfg=dict(filter_empty=True, min_size=1),
            ),
        ),
        val_dataloader=dict(
            batch_size=2,
            num_workers=4,
            persistent_workers=True,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=dict(
                type="CocoDataset",
                ann_file=str(val_json),
                data_prefix=dict(img=str(val_json.parent.parent / "images" / "val")),
                test_mode=True
            ),
        ),
        test_dataloader=dict(
            batch_size=2,
            num_workers=4,
            persistent_workers=True,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=dict(
                type="CocoDataset",
                ann_file=str(test_json),
                data_prefix=dict(img=str(test_json.parent.parent / "images" / "test")),
                test_mode=True
            ),
        ),

        # Preprocesador y escalado (mantener ratio)
        train_pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="RandomResize", scale=[(imgsz, imgsz)], keep_ratio=True),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PackDetInputs"),
        ],
        test_pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(imgsz, imgsz), keep_ratio=True),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="PackDetInputs"),
        ],

        # AdamW + grad clip
        optim_wrapper=dict(
            type="OptimWrapper",
            optimizer=dict(type="AdamW", lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05),
            clip_grad=dict(max_norm=1.0, norm_type=2),
        ),
        param_scheduler=[
            dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
            dict(type="CosineAnnealingLR", T_max=epochs, by_epoch=True),
        ],

        train_cfg=dict(max_epochs=epochs, val_interval=1),
        val_evaluator=dict(type="CocoMetric", ann_file=str(val_json), metric="bbox"),
        test_evaluator=dict(type="CocoMetric", ann_file=str(test_json), metric="bbox"),

        work_dir=str(workdir),

        default_hooks=dict(
            checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="coco/bbox_mAP"),
            logger=dict(type="LoggerHook", interval=50),
            param_scheduler=dict(type="ParamSchedulerHook"),
            timer=dict(type="IterTimerHook"),
            sampler_seed=dict(type="DistSamplerSeedHook"),
        ),

        visualizer=dict(type="DetLocalVisualizer"),
        env_cfg=dict(cudnn_benchmark=True),
        randomness=dict(seed=42, deterministic=False)
    ))

    cfg.model.data_preprocessor = dict(type="DetDataPreprocessor",
                                       mean=[123.675, 116.28, 103.53],
                                       std=[58.395, 57.12, 57.375],
                                       bgr_to_rgb=True,
                                       pad_size_divisor=32)
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug_yaml", default="augmentation_config.yaml")
    ap.add_argument("--qual_yaml", default="quality_config.yaml")
    ap.add_argument("--backbone", default="swin_t", choices=["swin_t", "resnext"])
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--workdir", default="work_dirs/deformable_detr")
    args = ap.parse_args()

    aug_cfg = load_yaml(args.aug_yaml)
    qual_cfg = load_yaml(args.qual_yaml)

    train_dir, train_json = find_train_final(aug_cfg)
    val_json, test_json = find_val_test(qual_cfg)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.backbone, args.imgsz, args.epochs,
                    train_dir, train_json, val_json, test_json, workdir)

    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == "__main__":
    main()
