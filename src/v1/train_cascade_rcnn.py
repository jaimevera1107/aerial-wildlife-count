#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_cascade_rcnn.py
MMDetection - Cascade R-CNN + FPN con backbone Swin o ResNeXt.

Requisitos:
 - MMDetection instalado y funcional (con MMCV/CUDA correctos)
 - pip install pyyaml

Ejecución:
python train_cascade_rcnn.py --backbone swin_t --imgsz 896 --epochs 48
"""

import argparse
import os
from pathlib import Path
import yaml

from mmengine.config import Config
from mmengine.runner import Runner

# ---------------------------
# Helpers
# ---------------------------
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
        raise FileNotFoundError("No encontré val/test JSON en quality_config.yaml (revisa 'splits').")
    # Asumimos imágenes en los mismos árboles de tu dataset original
    return val_json, test_json

def build_cfg(backbone: str, imgsz: int, epochs: int,
              train_img_dir: Path, train_json: Path,
              val_json: Path, test_json: Path,
              workdir: Path) -> Config:
    """
    Construye una Config de MMDetection programáticamente.
    """
    # Backbone & base
    if backbone.lower().startswith("swin"):
        base = [
            "mmdet::_base_/models/cascade-rcnn_swin-t-p4-w7_fpn.py",
            "mmdet::_base_/datasets/coco_instance.py",
            "mmdet::_base_/schedules/schedule_1x.py",
            "mmdet::_base_/default_runtime.py",
        ]
        pretrained = "mmdet://swin-tiny-p4-w7_..."  # MMDet resolver
    else:
        # ResNeXt-101 32x4d
        base = [
            "mmdet::_base_/models/cascade_rcnn_x101_32x4d_fpn.py",
            "mmdet::_base_/datasets/coco_instance.py",
            "mmdet::_base_/schedules/schedule_1x.py",
            "mmdet::_base_/default_runtime.py",
        ]
        pretrained = None

    cfg = Config(dict(
        _base_=base,
        default_scope="mmdet",

        dataset_type="CocoDataset",
        data_root=str(train_img_dir.parent),  # no imprescindible

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
                data_prefix=dict(img=str(val_json.parent.parent / "images" / "val")),  # ajusta si difiere
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

        # Aug (simple, la fuerte ya la hiciste en tu pipeline)
        train_pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="Resize", scale=(imgsz, imgsz), keep_ratio=True),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PackDetInputs"),
        ],
        test_pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", scale=(imgsz, imgsz), keep_ratio=True),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="PackDetInputs"),
        ],

        optim_wrapper=dict(
            type="OptimWrapper",
            optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001),
            clip_grad=dict(max_norm=0.5, norm_type=2),
        ),
        param_scheduler=[
            dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
            dict(type="MultiStepLR", begin=0, end=epochs, by_epoch=True, milestones=[int(epochs*0.67), int(epochs*0.89)], gamma=0.1),
        ],

        train_cfg=dict(max_epochs=epochs, val_interval=1),
        val_evaluator=dict(type="CocoMetric", ann_file=str(val_json), metric="bbox"),
        test_evaluator=dict(type="CocoMetric", ann_file=str(test_json), metric="bbox"),

        work_dir=str(workdir),
        load_from=pretrained,
        resume=False,

        visualizer=dict(type="DetLocalVisualizer"),
        default_hooks=dict(
            checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="coco/bbox_mAP"),
            logger=dict(type="LoggerHook", interval=50),
            param_scheduler=dict(type="ParamSchedulerHook"),
            timer=dict(type="IterTimerHook"),
            sampler_seed=dict(type="DistSamplerSeedHook"),
        ),

        env_cfg=dict(cudnn_benchmark=True),
        randomness=dict(seed=42, deterministic=False)
    ))

    # Ajuste de img_scale en data_preprocessor (algunos bases lo incluyen)
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
    ap.add_argument("--backbone", default="resnext", choices=["resnext", "swin_t"])
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--epochs", type=int, default=48)
    ap.add_argument("--workdir", default="work_dirs/cascade")
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
