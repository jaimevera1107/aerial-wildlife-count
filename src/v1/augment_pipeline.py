#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
augment_pipeline.py
Pipeline ejecutable de aumentación de datos con balanceo adaptativo.

Este script es la versión ejecutable del notebook augment.ipynb.
Proporciona aumentación avanzada de datasets con balanceo de clases,
rebalanceo adaptativo y generación de reportes.

Uso:
    python augment_pipeline.py --config augmentation_config.yaml --split train_joined
    python augment_pipeline.py --config augmentation_config.yaml --split train_joined --stages prop,rebalance
"""

import argparse
import json
import time
import logging
import warnings
import sys
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
import albumentations as A
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import cv2
import itertools
import threading
from concurrent.futures import as_completed
import math
import shutil
import random
import torch

warnings.filterwarnings("ignore", message="Got processor for bboxes, but no transform to process it")

class AugmentationPipeline:
    def __init__(self, config: dict, split: str = "train", logger=None):
        """
        Initialize configuration, directory structure, and parameters
        for the modular augmentation pipeline.
        """

        # ==========================================================
        # LOGGER SETUP
        # ==========================================================
        self.config = config
        self.split = split
        self.version = "v2.4.0"  # updated version marker

        log_cfg = config.get("logging", {})
        log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

        self.logger = logger or logging.getLogger(f"AugmentationPipeline.{split}")
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.propagate = False

        # ==========================================================
        # PATH CONFIGURATION
        # ==========================================================
        self.root_dir = Path(config.get("output_dir", "data/outputs")).resolve()
        self.mirror_dir = self.root_dir / "mirror_clean"

        paths_cfg = config.get("paths", {})

        # --- Base clean dataset ---
        self.clean_dir = Path(paths_cfg.get("clean_dir", self.mirror_dir / f"{split}_clean")).resolve()
        self.clean_json = Path(paths_cfg.get("clean_json", self.clean_dir / f"{split}_clean.json")).resolve()

        # --- JSON fallback detection ---
        if not self.clean_json.exists():
            fallback_candidates = [
                self.clean_dir / f"{split}_joined.json",
                self.clean_dir / f"{split}.json"
            ]
            found = next((p for p in fallback_candidates if p.exists()), None)
            if found:
                self.clean_json = found
                self.logger.info(f"[INIT] Using fallback JSON -> {found}")
            else:
                json_candidates = list(self.clean_dir.glob("*.json"))
                if len(json_candidates) == 1:
                    self.clean_json = json_candidates[0]
                    self.logger.info(f"[INIT] Auto-detected clean JSON -> {self.clean_json}")
                else:
                    raise FileNotFoundError(f"[INIT] Missing clean JSON file in {self.clean_dir}")

        if not self.clean_dir.exists():
            raise FileNotFoundError(f"[INIT] Missing clean image directory: {self.clean_dir}")

        # ==========================================================
        # DIRECTORY STRUCTURE (Standardized for all stages)
        # ==========================================================
        self.prop_dir = self.mirror_dir / f"{split}_prop"
        self.rebalance_dir = self.mirror_dir / f"{split}_rebalance_1"
        self.zoom_dir = self.mirror_dir / f"{split}_zoom"
        self.final_dir = self.mirror_dir / f"{split}_final"

        for d in [self.prop_dir, self.rebalance_dir, self.zoom_dir, self.final_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # --- Reports directory ---
        self.output_dir = self.root_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ==========================================================
        # JSON PATHS PER STAGE
        # ==========================================================
        self.prop_json = self.prop_dir / f"{split}_prop.json"
        self.rebalance_json = self.rebalance_dir / f"{split}_rebalance_1.json"
        self.zoom_json = self.zoom_dir / f"{split}_zoom.json"
        self.final_json = self.final_dir / f"{split}_final.json"

        # ==========================================================
        # AUGMENTATION PARAMETERS
        # ==========================================================
        aug_cfg = config.get("augmentations", {})

        def _to_float(val, default):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        def _to_int(val, default):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        self.mode = aug_cfg.get("mode", "none")
        self.proportion = _to_float(aug_cfg.get("proportion", 0.25), 0.25)
        self.tolerance = _to_float(aug_cfg.get("tolerance", 0.10), 0.10)
        self.max_repeats = _to_int(aug_cfg.get("max_repeats", 10), 10)
        self.min_unique_ratio = _to_float(aug_cfg.get("min_unique_ratio", 0.4), 0.4)
        self.max_boxes_per_image = _to_int(aug_cfg.get("max_boxes_per_image", 100), 100)
        self.min_images_per_class = _to_int(aug_cfg.get("min_images_per_class", 10), 10)
        self.transforms_cfg = aug_cfg.get("transforms", {})
        self.seed = _to_int(config.get("seed", 42), 42)
        self.num_workers = _to_int(config.get("num_workers", 4), 4)

        # ==========================================================
        # CLASS SETTINGS
        # ==========================================================
        self.classes = list(dict.fromkeys(config.get("classes", [])))  # remove duplicates, preserve order
        if not self.classes:
            raise ValueError("[INIT] No classes defined in configuration (config['classes']).")

        self.minority_strategy = aug_cfg.get("minority_strategy", "auto")  # "auto" or "bottom_n"
        self.bottom_n = _to_int(aug_cfg.get("bottom_n", 2), 2)

        if not (0 <= self.proportion <= 1):
            raise ValueError(f"[INIT] Invalid proportion value: {self.proportion}")
        if not (0 <= self.tolerance <= 1):
            raise ValueError(f"[INIT] Invalid tolerance value: {self.tolerance}")

        # ==========================================================
        # INTERNAL STATE
        # ==========================================================
        self.images = []
        self.annotations = []
        self.categories = []
        self.class_counts = {}
        self.pipeline = None
        self.sampling_plan = {}
        self.rebalance_plan = {}
        self.history = {}

        adaptive_flag = aug_cfg.get("adaptive_rebalance", False)
        self.logger.info(f"Adaptive rebalance={'ON' if adaptive_flag else 'OFF'}")

        # ==========================================================
        # FINAL VALIDATION LOG
        # ==========================================================
        self.logger.info("[INIT] Directory structure validated and configuration is consistent.")
        self.logger.info(f"[INIT] AugmentationPipeline ({self.version}) initialized for split='{split}'.")
        self.logger.info(
            f"Mode={self.mode} | Proportion={self.proportion} | Tolerance={self.tolerance} | "
            f"max_repeats={self.max_repeats} | min_unique_ratio={self.min_unique_ratio} | "
            f"max_boxes_per_image={self.max_boxes_per_image}"
        )

    def _filter_minority_images(self, anns: list[dict], cats: dict[int, str], minority_classes: list[str]) -> set[int]:
        """
        Identify and return image IDs that contain at least one annotation
        of any minority class. These are the only images eligible for the
        rebalance augmentation stage.
        """
        if not anns:
            self.logger.warning(f"[FILTER] ({self.split}) Empty annotation list. Returning empty set.")
            return set()
        if not cats:
            self.logger.warning(f"[FILTER] ({self.split}) Empty category mapping. Returning empty set.")
            return set()
        if not minority_classes:
            self.logger.info(f"[FILTER] ({self.split}) No minority classes provided. Nothing to filter.")
            return set()

        # --- Normalize class names (case-insensitive match) ---
        minority_norm = {c.lower().strip() for c in minority_classes}
        minority_cat_ids = {
            cid for cid, cname in cats.items()
            if cname and cname.lower().strip() in minority_norm
        }

        if not minority_cat_ids:
            self.logger.warning(f"[FILTER] ({self.split}) No matching category IDs found for minority classes.")
            return set()

        # --- Select eligible image IDs ---
        eligible_img_ids = {
            a["image_id"]
            for a in anns
            if "image_id" in a and a.get("category_id") in minority_cat_ids
        }

        # --- Reporting ---
        n_imgs = len(eligible_img_ids)
        n_min = len(minority_classes)

        if n_imgs == 0:
            self.logger.warning(f"[FILTER] ({self.split}) No eligible images found for {n_min} minority classes.")
        else:
            self.logger.info(
                f"[FILTER] ({self.split}) Eligible images for rebalance: {n_imgs} "
                f"(contain ≥1 of {n_min} minority classes)."
            )

        self.logger.debug(f"[FILTER] Minority classes ({n_min}): {minority_classes}")
        return eligible_img_ids

    def summarize_balance_from_json(self, json_path: Path) -> dict:
        """
        Compute class balance summary directly from a COCO-style JSON file.
        """
        if not Path(json_path).exists():
            raise FileNotFoundError(f"[SUMMARY] JSON file not found: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"[SUMMARY] Failed to read {json_path}: {e}")

        anns = data.get("annotations", [])
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        if not anns or not cats:
            raise ValueError("[SUMMARY] Invalid or empty dataset structure.")

        # Count per class
        counts = {cname: 0 for cname in cats.values()}
        for ann in anns:
            cname = cats.get(ann["category_id"])
            if cname:
                counts[cname] += 1

        total = sum(counts.values())
        if total == 0:
            raise RuntimeError("[SUMMARY] No annotations found in dataset.")

        # Stats
        max_c = max(counts.values())
        min_c = min(counts.values())
        mean_c = np.mean(list(counts.values()))
        std_c = np.std(list(counts.values()))
        ratio = round(max_c / max(min_c, 1), 2)
        cv = round(std_c / max(mean_c, 1e-9), 3)
        deviation = {k: round((v - mean_c) / mean_c, 3) for k, v in counts.items()}
        tol = float(getattr(self, "tolerance", 0.1))
        within_tol = np.mean(np.abs(list(deviation.values()))) <= tol

        # Detect minority / majority
        threshold = mean_c * (1 - tol)
        minority = [k for k, v in counts.items() if v < threshold]

        self.logger.info(f"[SUMMARY] Class balance summary for {json_path.name}:")
        self.logger.info(f"       -> Max={max_c} | Min={min_c} | Mean={mean_c:.1f} | Ratio={ratio:.2f}")
        self.logger.info(f"       -> Std={std_c:.1f} | CV={cv:.3f} | Within tol={within_tol}")

        for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
            delta = ((max_c - count) / max_c) * 100
            mark = "OK" if cls not in minority else "WARNING"
            self.logger.info(f"   - {cls:<15} {count:>6} Δ={delta:5.1f}% {mark}")

        return {
            "counts": counts,
            "max_count": max_c,
            "min_count": min_c,
            "mean_count": round(mean_c, 2),
            "std_count": round(std_c, 2),
            "cv": cv,
            "ratio": ratio,
            "minority_classes": minority,
            "within_tolerance": within_tol,
            "total_annotations": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def load_data(self) -> dict:
        """
        Load and validate COCO-style JSON with image and annotation metadata.
        Ensures internal consistency between image IDs and annotations.
        """
        if not self.clean_json.exists():
            raise FileNotFoundError(f"[LOAD] JSON not found: {self.clean_json}")
        try:
            with open(self.clean_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            size_kb = round(Path(self.clean_json).stat().st_size / 1024, 1)
            self.logger.info(f"[LOAD] Loaded JSON ({size_kb} KB) from {self.clean_json}")
        except json.JSONDecodeError as e:
            raise ValueError(f"[LOAD] Invalid JSON format: {e}")

        # --- Extract core fields ---
        self.images = data.get("images", [])
        self.annotations = data.get("annotations", [])
        self.categories = data.get("categories", [])

        if not self.images:
            self.logger.warning(f"[LOAD] No images found in JSON ({self.clean_json}).")
        if not self.categories:
            self.logger.warning(f"[LOAD] No categories found in JSON ({self.clean_json}).")

        # --- Validate internal links ---
        valid_ids = {img.get("id") for img in self.images if "id" in img}
        before = len(self.annotations)
        self.annotations = [a for a in self.annotations if a.get("image_id") in valid_ids]
        removed = before - len(self.annotations)

        # --- Build class maps ---
        self.id2class = {c["id"]: c["name"] for c in self.categories if "id" in c and "name" in c}
        self.class2id = {v: k for k, v in self.id2class.items()}

        if not self.id2class:
            raise ValueError("[LOAD] No valid categories with 'id' and 'name' fields found.")

        # --- Detect inconsistencies between config and JSON ---
        missing_in_cfg = set(self.id2class.values()) - set(self.classes)
        extra_in_cfg = set(self.classes) - set(self.id2class.values())
        if missing_in_cfg:
            self.logger.warning(f"[LOAD] Classes in JSON not listed in config: {missing_in_cfg}")
        if extra_in_cfg:
            self.logger.info(f"[LOAD] Classes in config not present in JSON: {extra_in_cfg}")

        # --- Count annotations per class ---
        self.class_counts = {cls: 0 for cls in self.classes}
        for ann in self.annotations:
            cname = self.id2class.get(ann.get("category_id"))
            if cname in self.class_counts:
                self.class_counts[cname] += 1
            else:
                self.logger.debug(f"[LOAD] Unknown category_id={ann.get('category_id')} found in annotations.")

        # --- Integrity checks ---
        unique_image_ids = len(valid_ids)
        unique_ann_ids = len({a["id"] for a in self.annotations if "id" in a})

        if unique_ann_ids < len(self.annotations):
            self.logger.warning("[LOAD] Duplicate annotation IDs detected.")
        if len(valid_ids) < len(self.images):
            self.logger.warning("[LOAD] Duplicate image IDs detected.")
        if removed > 0:
            self.logger.warning(f"[LOAD] Removed {removed} orphan annotations (invalid image references).")
            
        n_imgs = len(self.images)
        n_anns = len(self.annotations)
        n_cls = len(self.categories)
        self.logger.info(f"[LOAD] Split '{self.split}' loaded successfully -> {n_imgs} images, {n_anns} annotations, {n_cls} classes.")

        return {
            "images": n_imgs,
            "annotations": n_anns,
            "classes": n_cls,
            "unique_images": unique_image_ids,
            "unique_annotations": unique_ann_ids,
            "removed_orphans": removed
        }

    def analyze_class_distribution(self, plot: bool = True) -> dict | None:
        """
        Compute and optionally visualize annotation counts per class.
        Returns a summary dict or None if empty.
        """
        # --- Ensure data is loaded ---
        if not self.class_counts:
            self.logger.warning(f"[ANALYZE] Class counts not initialized. Reloading split='{self.split}'.")
            self.load_data()
            if not self.class_counts:
                return None

        total = sum(self.class_counts.values())
        if total == 0:
            self.logger.warning(f"[ANALYZE] No annotations for split='{self.split}'.")
            return None

        # --- Build arrays ---
        classes = list(self.class_counts.keys())
        counts = np.array([self.class_counts[c] for c in classes], dtype=int)
        percents = np.round(100 * counts / total, 2)

        # --- Sort descending for clarity ---
        order = np.argsort(-counts)
        classes = [classes[i] for i in order]
        counts = counts[order]
        percents = percents[order]
        
        # --- Compute stats for consistency with post-rebalance evaluation ---
        max_c = int(counts.max())
        min_c = int(counts.min())
        mean_c = float(np.mean(counts))
        ratio = round(max_c / max(min_c, 1), 2)
        tol = float(getattr(self, "tolerance", 0.1))
        deviation_mean = float(np.mean(np.abs((counts - mean_c) / mean_c)))
        within_tolerance = deviation_mean <= tol

        # --- Logging summary ---
        self.logger.info(f"[ANALYZE] Annotation distribution for split='{self.split}':")
        for c, n, p in zip(classes, counts, percents):
            self.logger.info(f"   - {c:<15}: {n:>6} ({p:5.2f}%)")
        self.logger.info(f"   Total annotations: {int(total)}")

        # --- Plot visualization ---
        if plot:
            try:
                sns.set(style="whitegrid", font_scale=0.9)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                palette = sns.color_palette("crest", len(classes))

                sns.barplot(x=classes, y=counts, ax=ax, palette=palette, legend=False)
                ax.set_ylim(0, max(counts) * 1.15)

                # Annotate counts and percentages
                offset = max(3, max(counts) * 0.02)
                for i, c in enumerate(classes):
                    ax.text(
                        i,
                        counts[i] + offset,
                        f"{counts[i]} ({percents[i]:.1f}%)",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="black"
                    )

                ax.set_title(f"Class Distribution - {self.split.upper()} (mean dev={deviation_mean:.3f}, tol={tol:.2f})", fontsize=12, fontweight="bold")
                ax.set_xlabel("Class")
                ax.set_ylabel("Number of Annotations")
                plt.xticks(rotation=25)
                plt.tight_layout()
                
                # Save plot instead of showing
                plot_path = self.output_dir / f"class_distribution_{self.split}_initial.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"[ANALYZE] Plot saved to: {plot_path}")

            except Exception as e:
                self.logger.warning(f"[ANALYZE] Plotting failed ({type(e).__name__}): {e}")

        return {
            "split": self.split,
            "classes": classes,
            "counts": counts.tolist(),
            "percents": percents.tolist(),
            "total": int(total),
            "max_count": max_c,
            "min_count": min_c,
            "mean_count": round(mean_c, 2),
            "ratio_max_min": ratio,
            "deviation_mean": round(deviation_mean, 4),
            "within_tolerance": within_tolerance,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def summarize_balance(self) -> dict:
        """
        Quantify imbalance ratios and identify minority/majority classes.
        Returns detailed per-class ratios, deviations, and global imbalance metrics.
        """
        # --- Ensure data ---
        if not self.class_counts:
            self.load_data()
        counts = self.class_counts
        if not counts:
            self.logger.warning(f"[BALANCE] No class counts found for split='{self.split}'.")
            return {}

        # --- Basic stats ---
        values = np.array(list(counts.values()), dtype=float)
        max_count = float(np.max(values))
        min_count = float(np.min(values))
        mean_count = float(np.mean(values))
        std_count = float(np.std(values))
        tol = float(self.tolerance)

        if mean_count == 0:
            self.logger.warning(f"[BALANCE] Empty dataset (mean count = 0) for split='{self.split}'.")
            return {}

        total_count = int(np.sum(values))
        cv = round(std_count / mean_count, 3)

        # --- Ratios and deviations ---
        ratios_to_max = {cls: round(cnt / max_count, 3) for cls, cnt in counts.items()}
        ratios_to_mean = {cls: round(cnt / mean_count, 3) for cls, cnt in counts.items()}
        deviation_signed = {cls: (cnt - mean_count) / mean_count for cls, cnt in counts.items()}
        deviation_abs = {cls: abs(v) for cls, v in deviation_signed.items()}

        # --- Determine minority and majority classes ---
        if getattr(self, "minority_strategy", "auto") == "bottom_n":
            if len(counts) <= self.bottom_n:
                self.logger.warning("[BALANCE] bottom_n strategy ignored (fewer classes than N).")
                minority, majority = [], []
            else:
                sorted_classes = sorted(counts.items(), key=lambda x: x[1])
                minority = [cls for cls, _ in sorted_classes[: self.bottom_n]]
                majority = [cls for cls, _ in sorted(counts.items(), key=lambda x: -x[1])[: self.bottom_n]]
                self.logger.info(f"[BALANCE] Using fixed bottom_n strategy ({self.bottom_n}).")
        else:
            minority = [cls for cls, dev in deviation_abs.items() if counts[cls] < mean_count and dev > tol]
            majority = [cls for cls, dev in deviation_abs.items() if counts[cls] > mean_count and dev > tol]

        imbalance_ratio = round(max_count / max(min_count, 1), 2)
        balanced_within_tol = not (minority or majority)

        self.logger.info(
            f"[BALANCE] Split='{self.split}' | tol={tol:.2f} | ratio={imbalance_ratio:.2f} | CV={cv:.3f}"
        )
        self.logger.info(
            f"[BALANCE] Max={max_count:.0f} ({max(counts, key=counts.get)}) | "
            f"Min={min_count:.0f} ({min(counts, key=counts.get)}) | Mean={mean_count:.1f}"
        )

        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            rmax = ratios_to_max[cls]
            rmean = ratios_to_mean[cls]
            devp = deviation_signed[cls] * 100
            mark = "MINORITY" if cls in minority else ("MAJORITY" if cls in majority else "")
            self.logger.info(
                f"   {cls:<15}: count={cnt:<6} | r_max={rmax:.3f} | r_mean={rmean:.3f} | Δ={devp:6.1f}% {mark}"
            )

        if minority:
            self.logger.info(f"[BALANCE] Minority ({len(minority)}): {minority}")
        if majority:
            self.logger.info(f"[BALANCE] Majority ({len(majority)}): {majority}")
        if balanced_within_tol:
            self.logger.info("[BALANCE] Dataset appears balanced within tolerance range.")

        return {
            "split": self.split,
            "total_annotations": total_count,
            "max_class": max(counts, key=counts.get),
            "min_class": min(counts, key=counts.get),
            "max_count": int(max_count),
            "min_count": int(min_count),
            "mean_count": round(mean_count, 2),
            "std_count": round(std_count, 2),
            "cv": cv,
            "ratios_to_max": ratios_to_max,
            "ratios_to_mean": ratios_to_mean,
            "deviation_signed": deviation_signed,
            "minority_classes": minority,
            "majority_classes": majority,
            "imbalance_ratio": imbalance_ratio,
            "balanced_within_tol": balanced_within_tol,
            "tolerance": tol
        }

    def set_seed_everywhere(self) -> dict:
        """
        Fix global random seeds across all supported modules for full reproducibility.

        Ensures deterministic behavior for:
        - Python's random module
        - NumPy random generator
        - PyTorch (CPU and CUDA, if available)
        """

        # ==========================================================
        # 🔹 Validate and normalize seed
        # ==========================================================
        try:
            self.seed = int(self.seed)
        except Exception:
            self.seed = 42
            if hasattr(self, "logger") and self.logger:
                self.logger.warning("[SETUP] Invalid seed type detected, fallback to 42.")

        # ==========================================================
        # 🔹 Set Python and NumPy seeds
        # ==========================================================
        random.seed(self.seed)
        np.random.seed(self.seed)

        cuda_available = False

        # ==========================================================
        # 🔹 PyTorch seed handling
        # ==========================================================
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    # Compatible fallback for older torch versions
                    pass

                cuda_available = True
        except ModuleNotFoundError:
            if hasattr(self, "logger") and self.logger:
                self.logger.debug("[SETUP] PyTorch not installed — skipping CUDA seeding.")
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.warning(f"[SETUP] Error while seeding PyTorch: {e}")

        # ==========================================================
        # 🔹 Logging and return
        # ==========================================================
        if hasattr(self, "logger") and self.logger:
            self.logger.info(f"[SETUP] Global seed fixed → {self.seed} | CUDA available: {cuda_available}")

        return {"seed": self.seed, "cuda_available": cuda_available}

    def prepare_transforms(self):
        """
        Dynamically build Albumentations pipeline from configuration.
        Each transform inside the pipeline is applied independently according to
        its individual probability `p`. The whole pipeline runs once per image
        (ReplayCompose has p=1.0).
        """
        cfg_aug = self.config.get("augmentations", {})

        if not cfg_aug.get("enabled", False):
            self.logger.info("[AUGMENT] Augmentations disabled in configuration.")
            self.pipeline = None
            return None

        transforms_dict = cfg_aug.get("transforms", {})
        if not transforms_dict:
            raise ValueError("[AUGMENT] No transforms defined in config['augmentations']['transforms'].")

        aug_list = []
        for name, params in transforms_dict.items():
            p = float(params.get("p", 0.5))  # individual transform probability

            try:
                transform_cls = getattr(A, name, None)
                if transform_cls is None:
                    self.logger.warning(f"[AUGMENT] Unknown transform '{name}' ignored.")
                    continue

                # Default border mode for Rotate
                if name == "Rotate" and "border_mode" not in params:
                    params["border_mode"] = cv2.BORDER_REFLECT_101

                # Filter out invalid keys
                valid_params = {k: v for k, v in params.items() if k != "p"}

                transform = transform_cls(**valid_params, p=p)
                aug_list.append(transform)

            except Exception as e:
                self.logger.warning(f"[AUGMENT] Error initializing '{name}': {e}")
                continue

        if not aug_list:
            raise ValueError("[AUGMENT] No valid transforms found after parsing configuration.")

        try:
            aug_blocks = []
            # Always ensure at least one transform is applied
            aug_blocks.append(A.OneOf(aug_list, p=1.0))
            # Optional: apply a second random combination (if SomeOf is supported)
            try:
                aug_blocks.append(A.SomeOf(aug_list, n=2, replace=False, p=0.3))
            except Exception:
                self.logger.warning("[AUGMENT] A.SomeOf not supported in this Albumentations version. Skipping secondary block.")

            self.pipeline = A.ReplayCompose(
                aug_blocks,
                bbox_params=A.BboxParams(
                    format="coco",
                    label_fields=["category_ids"],
                    min_visibility=0.25  # discard boxes mostly out of frame
                ),p=1.0  # pipeline always executed, internal transforms decide individually
            )
        except Exception as e:
            raise RuntimeError(f"[AUGMENT] Failed to build Albumentations pipeline: {e}")

        used_transforms = [t.__class__.__name__ for t in aug_list]
        self.logger.info(f"[AUGMENT] Albumentations pipeline created successfully.")
        self.logger.info(f"          Includes {len(aug_list)} transforms: {', '.join(used_transforms)}")

        avg_prob = np.mean([t.p for t in aug_list])
        self.logger.info(f"[AUGMENT] Expected average of {len(aug_list)*avg_prob:.2f} transforms applied per image (stochastically).")

        return self.pipeline

    def apply_global_proportional_augmentation(self) -> dict:
        """
        Apply proportional augmentation to all images in the dataset.
        This stage creates a baseline augmented dataset before rebalancing.
        """
        if not self.pipeline:
            self.logger.warning("[PROP] No augmentation pipeline available. Skipping proportional augmentation.")
            return {"status": "skipped", "reason": "no_pipeline"}

        self.logger.info(f"[PROP] Starting proportional augmentation for split='{self.split}'...")
        
        # Calculate how many images to augment
        total_images = len(self.images)
        num_to_augment = max(1, int(total_images * self.proportion))
        
        self.logger.info(f"[PROP] Will augment {num_to_augment}/{total_images} images ({self.proportion*100:.1f}%)")

        # Select images to augment (random sampling)
        np.random.seed(self.seed)
        selected_indices = np.random.choice(total_images, size=num_to_augment, replace=False)
        
        # Prepare output data structures
        augmented_images = []
        augmented_annotations = []
        next_img_id = max([img["id"] for img in self.images]) + 1
        next_ann_id = max([ann["id"] for ann in self.annotations]) + 1

        # Group annotations by image
        anns_by_img = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)

        def _augment_image(img_id, cls_name):
            """Augment a single image and return new data."""
            try:
                # Find image data
                img_data = next((img for img in self.images if img["id"] == img_id), None)
                if not img_data:
                    return None

                # Load image
                img_path = self.clean_dir / img_data["file_name"]
                if not img_path.exists():
                    self.logger.warning(f"[PROP] Image not found: {img_path}")
                    return None

                image = cv2.imread(str(img_path))
                if image is None:
                    self.logger.warning(f"[PROP] Failed to load image: {img_path}")
                    return None

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Get annotations for this image
                img_anns = anns_by_img.get(img_id, [])
                if not img_anns:
                    self.logger.warning(f"[PROP] No annotations for image {img_id}")
                    return None

                # Prepare bboxes and category_ids
                bboxes = []
                category_ids = []
                for ann in img_anns:
                    bbox = ann["bbox"]
                    if len(bbox) == 4:
                        bboxes.append(bbox)
                        category_ids.append(ann["category_id"])

                if not bboxes:
                    self.logger.warning(f"[PROP] No valid bboxes for image {img_id}")
                    return None

                # Apply augmentation
                try:
                    augmented = self.pipeline(image=image, bboxes=bboxes, category_ids=category_ids)
                    aug_image = augmented["image"]
                    aug_bboxes = augmented["bboxes"]
                    aug_category_ids = augmented["category_ids"]
                except Exception as e:
                    self.logger.warning(f"[PROP] Augmentation failed for image {img_id}: {e}")
                    return None

                # Generate new IDs
                new_img_id = next_img_id
                next_img_id += 1

                # Create new image record
                new_img_data = {
                    "id": new_img_id,
                    "file_name": f"aug_{img_data['id']}_{cls_name}_{uuid4().hex[:8]}.jpg",
                    "width": aug_image.shape[1],
                    "height": aug_image.shape[0]
                }

                # Save augmented image
                output_path = self.prop_dir / new_img_data["file_name"]
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), aug_image_bgr)

                # Create new annotations
                new_anns = []
                for bbox, cat_id in zip(aug_bboxes, aug_category_ids):
                    new_ann = {
                        "id": next_ann_id,
                        "image_id": new_img_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    }
                    new_anns.append(new_ann)
                    next_ann_id += 1

                return {
                    "image": new_img_data,
                    "annotations": new_anns
                }

            except Exception as e:
                self.logger.error(f"[PROP] Error augmenting image {img_id}: {e}")
                return None

        # Process images in parallel
        augmented_count = 0
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for idx in selected_indices:
                img_data = self.images[idx]
                img_id = img_data["id"]
                # Use a representative class name for this image
                img_anns = anns_by_img.get(img_id, [])
                if img_anns:
                    cls_name = self.id2class.get(img_anns[0]["category_id"], "unknown")
                else:
                    cls_name = "unknown"
                
                future = executor.submit(_augment_image, img_id, cls_name)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Proportional augmentation"):
                result = future.result()
                if result:
                    augmented_images.append(result["image"])
                    augmented_annotations.extend(result["annotations"])
                    augmented_count += 1

        # Combine original and augmented data
        final_images = self.images + augmented_images
        final_annotations = self.annotations + augmented_annotations

        # Save results
        output_data = {
            "images": final_images,
            "annotations": final_annotations,
            "categories": self.categories
        }

        with open(self.prop_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        self.logger.info(f"[PROP] Proportional augmentation completed:")
        self.logger.info(f"       Original: {len(self.images)} images, {len(self.annotations)} annotations")
        self.logger.info(f"       Augmented: {augmented_count} images, {len(augmented_annotations)} annotations")
        self.logger.info(f"       Final: {len(final_images)} images, {len(final_annotations)} annotations")
        self.logger.info(f"       Saved to: {self.prop_json}")

        return {
            "split": self.split,
            "augmented_images": len(augmented_images),
            "augmented_annotations": len(augmented_annotations),
            "output_dir": str(self.prop_dir),
            "json_path": str(self.prop_json),
            "duration_sec": 0.0,  # Could be tracked if needed
            "skipped": False,
        }

    def define_rebalance_strategy(self) -> dict:
        """
        Analyze class distribution and define rebalancing strategy.
        """
        self.logger.info(f"[REBALANCE] Defining rebalancing strategy for split='{self.split}'...")
        
        # Get current balance
        balance_info = self.summarize_balance()
        minority_classes = balance_info.get("minority_classes", [])
        
        if not minority_classes:
            self.logger.info("[REBALANCE] No minority classes detected. Rebalancing not needed.")
            return {"strategy": "none", "minority_classes": []}
        
        self.logger.info(f"[REBALANCE] Minority classes identified: {minority_classes}")
        
        # Calculate rebalancing needs
        max_count = balance_info["max_count"]
        minority_counts = {cls: self.class_counts[cls] for cls in minority_classes}
        
        rebalance_plan = {}
        for cls in minority_classes:
            current_count = minority_counts[cls]
            target_count = int(max_count * (1 - self.tolerance))
            needed = max(0, target_count - current_count)
            
            if needed > 0:
                rebalance_plan[cls] = {
                    "current": current_count,
                    "target": target_count,
                    "needed": needed,
                    "multiplier": target_count / current_count if current_count > 0 else 1.0
                }
        
        self.rebalance_plan = rebalance_plan
        self.logger.info(f"[REBALANCE] Rebalancing plan created for {len(rebalance_plan)} classes")
        
        return {
            "strategy": "over_sampling",
            "minority_classes": minority_classes,
            "rebalance_plan": rebalance_plan
        }

    def plan_rebalance_sampling(self) -> dict:
        """
        Plan the sampling strategy for rebalancing augmentation.
        """
        if not self.rebalance_plan:
            self.logger.warning("[SAMPLING] No rebalance plan available. Run define_rebalance_strategy first.")
            return {}
        
        self.logger.info(f"[SAMPLING] Planning rebalance sampling for split='{self.split}'...")
        
        # Get images containing minority classes
        minority_classes = list(self.rebalance_plan.keys())
        minority_cat_ids = {self.class2id[cls] for cls in minority_classes if cls in self.class2id}
        
        eligible_images = set()
        for ann in self.annotations:
            if ann["category_id"] in minority_cat_ids:
                eligible_images.add(ann["image_id"])
        
        self.logger.info(f"[SAMPLING] Found {len(eligible_images)} images containing minority classes")
        
        # Plan sampling for each minority class
        sampling_plan = {}
        for cls, plan in self.rebalance_plan.items():
            needed = plan["needed"]
            cat_id = self.class2id.get(cls)
            if not cat_id:
                continue
            
            # Find images with this class
            class_images = set()
            for ann in self.annotations:
                if ann["category_id"] == cat_id:
                    class_images.add(ann["image_id"])
            
            if not class_images:
                continue
            
            # Calculate how many times to augment each image
            images_needed = min(len(class_images), needed)
            augmentations_per_image = max(1, needed // images_needed)
            remaining = needed % images_needed
            
            sampling_plan[cls] = {
                "class_images": list(class_images),
                "augmentations_per_image": augmentations_per_image,
                "remaining_augmentations": remaining,
                "total_needed": needed
            }
        
        self.sampling_plan = sampling_plan
        self.logger.info(f"[SAMPLING] Sampling plan created for {len(sampling_plan)} classes")
        
        return sampling_plan

    def apply_rebalance_augmentation(self, source: str = "prop", iteration: int = 1) -> dict:
        """
        Apply rebalancing augmentation based on the sampling plan.
        """
        if not self.sampling_plan:
            self.logger.warning("[REBALANCE] No sampling plan available. Run plan_rebalance_sampling first.")
            return {"status": "skipped", "reason": "no_sampling_plan"}
        
        self.logger.info(f"[REBALANCE] Starting rebalance augmentation (iteration {iteration})...")
        
        # Load source data
        if source == "prop" and self.prop_json.exists():
            with open(self.prop_json, "r", encoding="utf-8") as f:
                source_data = json.load(f)
            source_images = source_data["images"]
            source_annotations = source_data["annotations"]
        else:
            source_images = self.images
            source_annotations = self.annotations
        
        # Create image lookup
        img_lookup = {img["id"]: img for img in source_images}
        
        # Group annotations by image
        anns_by_img = {}
        for ann in source_annotations:
            img_id = ann["image_id"]
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)
        
        # Prepare output data
        rebalanced_images = []
        rebalanced_annotations = []
        next_img_id = max([img["id"] for img in source_images]) + 1
        next_ann_id = max([ann["id"] for ann in source_annotations]) + 1
        
        def _augment_image(img_id, cls_name="mixed"):
            """Augment a single image for rebalancing."""
            try:
                img_data = img_lookup.get(img_id)
                if not img_data:
                    return None
                
                # Load image
                img_path = self.clean_dir / img_data["file_name"]
                if not img_path.exists():
                    self.logger.warning(f"[REBALANCE] Image not found: {img_path}")
                    return None
                
                image = cv2.imread(str(img_path))
                if image is None:
                    return None
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get annotations
                img_anns = anns_by_img.get(img_id, [])
                if not img_anns:
                    return None
                
                bboxes = []
                category_ids = []
                for ann in img_anns:
                    bbox = ann["bbox"]
                    if len(bbox) == 4:
                        bboxes.append(bbox)
                        category_ids.append(ann["category_id"])
                
                if not bboxes:
                    return None
                
                # Apply augmentation
                try:
                    augmented = self.pipeline(image=image, bboxes=bboxes, category_ids=category_ids)
                    aug_image = augmented["image"]
                    aug_bboxes = augmented["bboxes"]
                    aug_category_ids = augmented["category_ids"]
                except Exception as e:
                    self.logger.warning(f"[REBALANCE] Augmentation failed: {e}")
                    return None
                
                # Create new image record
                new_img_id = next_img_id
                next_img_id += 1
                
                new_img_data = {
                    "id": new_img_id,
                    "file_name": f"rebal_{img_data['id']}_{cls_name}_{uuid4().hex[:8]}.jpg",
                    "width": aug_image.shape[1],
                    "height": aug_image.shape[0]
                }
                
                # Save image
                output_path = self.rebalance_dir / new_img_data["file_name"]
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), aug_image_bgr)
                
                # Create annotations
                new_anns = []
                for bbox, cat_id in zip(aug_bboxes, aug_category_ids):
                    new_ann = {
                        "id": next_ann_id,
                        "image_id": new_img_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    }
                    new_anns.append(new_ann)
                    next_ann_id += 1
                
                return {
                    "image": new_img_data,
                    "annotations": new_anns
                }
                
            except Exception as e:
                self.logger.error(f"[REBALANCE] Error: {e}")
                return None
        
        # Process each class
        total_augmented = 0
        for cls, plan in self.sampling_plan.items():
            self.logger.info(f"[REBALANCE] Processing class '{cls}' - need {plan['total_needed']} augmentations")
            
            class_images = plan["class_images"]
            augmentations_per_image = plan["augmentations_per_image"]
            remaining = plan["remaining_augmentations"]
            
            # Augment images
            augmented_count = 0
            for i, img_id in enumerate(class_images):
                if augmented_count >= plan["total_needed"]:
                    break
                
                # Calculate how many times to augment this image
                times_to_augment = augmentations_per_image
                if i < remaining:
                    times_to_augment += 1
                
                for _ in range(times_to_augment):
                    if augmented_count >= plan["total_needed"]:
                        break
                    
                    result = _augment_image(img_id, cls)
                    if result:
                        rebalanced_images.append(result["image"])
                        rebalanced_annotations.extend(result["annotations"])
                        augmented_count += 1
                        total_augmented += 1
            
            self.logger.info(f"[REBALANCE] Class '{cls}': {augmented_count}/{plan['total_needed']} augmentations completed")
        
        # Combine with source data
        final_images = source_images + rebalanced_images
        final_annotations = source_annotations + rebalanced_annotations
        
        # Save results
        output_data = {
            "images": final_images,
            "annotations": final_annotations,
            "categories": self.categories
        }
        
        with open(self.rebalance_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"[REBALANCE] Rebalance augmentation completed:")
        self.logger.info(f"       Augmented: {total_augmented} images")
        self.logger.info(f"       Final: {len(final_images)} images, {len(final_annotations)} annotations")
        self.logger.info(f"       Saved to: {self.rebalance_json}")
        
        return {
            "split": self.split,
            "augmented_images": total_augmented,
            "augmented_annotations": len(rebalanced_annotations),
            "output_dir": str(self.rebalance_dir),
            "json_path": str(self.rebalance_json),
            "iteration": iteration
        }

    def apply_minority_focus_zoom(self) -> dict:
        """
        Apply minority focus zoom augmentation.
        """
        self.logger.info(f"[ZOOM] Starting minority focus zoom for split='{self.split}'...")
        
        # Get minority classes from rebalance plan
        if not self.rebalance_plan:
            self.logger.warning("[ZOOM] No rebalance plan available. Skipping focus zoom.")
            return {"status": "skipped", "reason": "no_rebalance_plan"}
        
        minority_classes = list(self.rebalance_plan.keys())
        if not minority_classes:
            self.logger.info("[ZOOM] No minority classes found. Skipping focus zoom.")
            return {"status": "skipped", "reason": "no_minority_classes"}
        
        # Load source data (use rebalanced if available, otherwise proportional)
        if self.rebalance_json.exists():
            with open(self.rebalance_json, "r", encoding="utf-8") as f:
                source_data = json.load(f)
            source_images = source_data["images"]
            source_annotations = source_data["annotations"]
        elif self.prop_json.exists():
            with open(self.prop_json, "r", encoding="utf-8") as f:
                source_data = json.load(f)
            source_images = source_data["images"]
            source_annotations = source_data["annotations"]
        else:
            source_images = self.images
            source_annotations = self.annotations
        
        # Calculate zoom parameters
        focus_reinforce_prop = self.config.get("augmentations", {}).get("focus_reinforce_prop", 0.08)
        focus_zoom_scale = self.config.get("augmentations", {}).get("focus_zoom_scale", 1.5)
        focus_zoom_offset = self.config.get("augmentations", {}).get("focus_zoom_offset", 0.2)
        
        num_to_zoom = max(1, int(len(source_images) * focus_reinforce_prop))
        
        self.logger.info(f"[ZOOM] Will apply focus zoom to {num_to_zoom} images")
        
        # Find images with minority classes
        minority_cat_ids = {self.class2id[cls] for cls in minority_classes if cls in self.class2id}
        minority_images = set()
        
        for ann in source_annotations:
            if ann["category_id"] in minority_cat_ids:
                minority_images.add(ann["image_id"])
        
        minority_images = list(minority_images)
        if len(minority_images) < num_to_zoom:
            num_to_zoom = len(minority_images)
        
        # Select images to zoom
        np.random.seed(self.seed)
        selected_indices = np.random.choice(len(minority_images), size=num_to_zoom, replace=False)
        selected_img_ids = [minority_images[i] for i in selected_indices]
        
        # Create image lookup
        img_lookup = {img["id"]: img for img in source_images}
        
        # Group annotations by image
        anns_by_img = {}
        for ann in source_annotations:
            img_id = ann["image_id"]
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)
        
        # Prepare output data
        zoomed_images = []
        zoomed_annotations = []
        next_img_id = max([img["id"] for img in source_images]) + 1
        next_ann_id = max([ann["id"] for ann in source_annotations]) + 1
        
        def _zoom_image(img_id):
            """Apply focus zoom to a single image."""
            try:
                img_data = img_lookup.get(img_id)
                if not img_data:
                    return None
                
                # Load image
                img_path = self.clean_dir / img_data["file_name"]
                if not img_path.exists():
                    return None
                
                image = cv2.imread(str(img_path))
                if image is None:
                    return None
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                
                # Calculate zoom parameters
                zoom_scale = focus_zoom_scale
                offset_x = int(w * focus_zoom_offset * np.random.uniform(-1, 1))
                offset_y = int(h * focus_zoom_offset * np.random.uniform(-1, 1))
                
                # Calculate crop region
                new_w = int(w / zoom_scale)
                new_h = int(h / zoom_scale)
                
                center_x = w // 2 + offset_x
                center_y = h // 2 + offset_y
                
                x1 = max(0, center_x - new_w // 2)
                y1 = max(0, center_y - new_h // 2)
                x2 = min(w, x1 + new_w)
                y2 = min(h, y1 + new_h)
                
                # Crop and resize
                cropped = image[y1:y2, x1:x2]
                zoomed = cv2.resize(cropped, (w, h))
                
                # Get annotations and adjust bboxes
                img_anns = anns_by_img.get(img_id, [])
                adjusted_anns = []
                
                for ann in img_anns:
                    bbox = ann["bbox"]
                    x, y, bw, bh = bbox
                    
                    # Adjust bbox coordinates for crop
                    new_x = (x - x1) * (w / (x2 - x1))
                    new_y = (y - y1) * (h / (y2 - y1))
                    new_w = bw * (w / (x2 - x1))
                    new_h = bh * (h / (y2 - y1))
                    
                    # Check if bbox is still valid
                    if new_x >= 0 and new_y >= 0 and new_x + new_w <= w and new_y + new_h <= h:
                        adjusted_anns.append({
                            "id": next_ann_id,
                            "image_id": next_img_id,
                            "category_id": ann["category_id"],
                            "bbox": [new_x, new_y, new_w, new_h],
                            "area": new_w * new_h,
                            "iscrowd": 0
                        })
                        next_ann_id += 1
                
                if not adjusted_anns:
                    return None
                
                # Create new image record
                new_img_data = {
                    "id": next_img_id,
                    "file_name": f"zoom_{img_data['id']}_{uuid4().hex[:8]}.jpg",
                    "width": w,
                    "height": h
                }
                next_img_id += 1
                
                # Save zoomed image
                output_path = self.zoom_dir / new_img_data["file_name"]
                zoomed_bgr = cv2.cvtColor(zoomed, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), zoomed_bgr)
                
                return {
                    "image": new_img_data,
                    "annotations": adjusted_anns
                }
                
            except Exception as e:
                self.logger.error(f"[ZOOM] Error: {e}")
                return None
        
        # Process images
        zoomed_count = 0
        for img_id in selected_img_ids:
            result = _zoom_image(img_id)
            if result:
                zoomed_images.append(result["image"])
                zoomed_annotations.extend(result["annotations"])
                zoomed_count += 1
        
        # Combine with source data
        final_images = source_images + zoomed_images
        final_annotations = source_annotations + zoomed_annotations
        
        # Save results
        output_data = {
            "images": final_images,
            "annotations": final_annotations,
            "categories": self.categories
        }
        
        with open(self.zoom_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"[ZOOM] Focus zoom completed:")
        self.logger.info(f"       Zoomed: {zoomed_count} images")
        self.logger.info(f"       Final: {len(final_images)} images, {len(final_annotations)} annotations")
        self.logger.info(f"       Saved to: {self.zoom_json}")
        
        return {
            "split": self.split,
            "zoomed_images": zoomed_count,
            "zoomed_annotations": len(zoomed_annotations),
            "output_dir": str(self.zoom_dir),
            "json_path": str(self.zoom_json)
        }

    def merge_augmented_splits(self, reindex: bool = True, clean_previous: bool = True) -> dict:
        """
        Merge all augmented stages into a final dataset.
        """
        self.logger.info(f"[MERGE] Starting merge of augmented splits for '{self.split}'...")
        
        # Determine which stages to merge
        stages_to_merge = []
        if self.zoom_json.exists():
            stages_to_merge.append(("zoom", self.zoom_json))
        elif self.rebalance_json.exists():
            stages_to_merge.append(("rebalance", self.rebalance_json))
        elif self.prop_json.exists():
            stages_to_merge.append(("prop", self.prop_json))
        else:
            # Use original data
            stages_to_merge.append(("original", None))
        
        self.logger.info(f"[MERGE] Merging stages: {[s[0] for s in stages_to_merge]}")
        
        # Load final stage data
        final_stage = stages_to_merge[-1][0]
        if final_stage == "original":
            final_images = self.images
            final_annotations = self.annotations
        else:
            final_json = stages_to_merge[-1][1]
            with open(final_json, "r", encoding="utf-8") as f:
                final_data = json.load(f)
            final_images = final_data["images"]
            final_annotations = final_data["annotations"]
        
        # Reindex if requested
        if reindex:
            self.logger.info("[MERGE] Reindexing IDs...")
            
            # Reindex images
            img_id_map = {}
            for i, img in enumerate(final_images):
                old_id = img["id"]
                new_id = i + 1
                img_id_map[old_id] = new_id
                img["id"] = new_id
            
            # Reindex annotations
            for i, ann in enumerate(final_annotations):
                ann["id"] = i + 1
                ann["image_id"] = img_id_map.get(ann["image_id"], ann["image_id"])
        
        # Save final dataset
        final_data = {
            "images": final_images,
            "annotations": final_annotations,
            "categories": self.categories
        }
        
        with open(self.final_json, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2)
        
        # Copy images to final directory
        final_img_dir = self.final_dir / "images"
        final_img_dir.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        for img in final_images:
            # Find source image
            source_path = None
            for stage_name, stage_json in stages_to_merge:
                if stage_name == "original":
                    source_path = self.clean_dir / img["file_name"]
                else:
                    stage_dir = getattr(self, f"{stage_name}_dir")
                    source_path = stage_dir / img["file_name"]
                
                if source_path.exists():
                    break
            
            if source_path and source_path.exists():
                dest_path = final_img_dir / img["file_name"]
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                except Exception as e:
                    self.logger.warning(f"[MERGE] Failed to copy {source_path}: {e}")
        
        self.logger.info(f"[MERGE] Merge completed:")
        self.logger.info(f"       Final: {len(final_images)} images, {len(final_annotations)} annotations")
        self.logger.info(f"       Images copied: {copied_count}/{len(final_images)}")
        self.logger.info(f"       Saved to: {self.final_json}")
        
        return {
            "split": self.split,
            "final_images": len(final_images),
            "final_annotations": len(final_annotations),
            "images_copied": copied_count,
            "output_dir": str(self.final_dir),
            "json_path": str(self.final_json),
            "stages_merged": [s[0] for s in stages_to_merge]
        }

    def run(self) -> dict:
        """
        Execute the complete modular augmentation pipeline for the given split.
        Each stage generates its own dataset folder and JSON file before
        final consolidation and verification.
        """
        start_time = time.time()
        self.set_seed_everywhere()
        self.logger.info(f"[RUN] Starting augmentation pipeline for split='{self.split}'...")

        try:
            phase_times = {}

            # ==========================================================
            # 1️⃣ Load data & analyze distribution
            # ==========================================================
            t0 = time.time()
            self.load_data()
            self.analyze_class_distribution(plot=False)
            phase_times["data_analysis_sec"] = round(time.time() - t0, 2)

            # ==========================================================
            # 2️⃣ Prepare Albumentations transforms
            # ==========================================================
            t0 = time.time()
            self.prepare_transforms()
            phase_times["transform_setup_sec"] = round(time.time() - t0, 2)

            # ==========================================================
            # 3️⃣ Global proportional augmentation
            # ==========================================================
            t0 = time.time()
            if self.prop_json.exists():
                self.logger.info("[RUN] Found existing proportional JSON. Skipping regeneration.")
                with open(self.prop_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prop_report = {
                    "split": self.split,
                    "augmented_images": len(data.get("images", [])),
                    "augmented_annotations": len(data.get("annotations", [])),
                    "output_dir": str(self.prop_dir),
                    "json_path": str(self.prop_json),
                    "duration_sec": 0.0,
                    "skipped": True,
                }
            else:
                prop_report = self.apply_global_proportional_augmentation()
            phase_times["proportional_aug_sec"] = round(time.time() - t0, 2)

            # ==========================================================
            # 4️⃣ Rebalance (adaptive or single-pass)
            # ==========================================================
            reb_report = {"rebalance_applied": False}
            if self.mode in ["over", "under"]:
                t0 = time.time()
                self.define_rebalance_strategy()
                aug_cfg = self.config.get("augmentations", {})
                adaptive_enabled = bool(aug_cfg.get("adaptive_rebalance", False))

                if adaptive_enabled and self.mode == "over":
                    self.logger.info("[RUN] Adaptive image-level rebalance enabled.")
                    # For now, use single-pass rebalance
                    self.plan_rebalance_sampling()
                    reb_report = self.apply_rebalance_augmentation()
                else:
                    self.logger.info("[RUN] Single-pass rebalance mode activated.")
                    self.plan_rebalance_sampling()
                    reb_report = self.apply_rebalance_augmentation()

                phase_times["rebalance_aug_sec"] = round(time.time() - t0, 2)
            else:
                self.logger.info(f"[RUN] Rebalance skipped (mode='{self.mode}').")
                phase_times["rebalance_aug_sec"] = 0.0

            # ==========================================================
            # 5️⃣ Minority focus zoom (optional)
            # ==========================================================
            t0 = time.time()
            zoom_report = {"focus_zoom_applied": False, "generated": 0}

            try:
                zoom_report = self.apply_minority_focus_zoom()
                gen = zoom_report.get("zoomed_images", 0)
                self.logger.info(f"[RUN] Minority zoom completed → {gen} new images.")
            except Exception as e:
                self.logger.warning(f"[RUN] Focus zoom skipped: {e}")
                zoom_report = {"focus_zoom_applied": False, "zoomed_images": 0, "error": str(e)}

            phase_times["focus_zoom_sec"] = round(time.time() - t0, 2)

            # ==========================================================
            # 6️⃣ Merge + final verification
            # ==========================================================
            t0 = time.time()
            merge_report = self.merge_augmented_splits(reindex=True, clean_previous=True)
            phase_times["merge_verify_sec"] = round(time.time() - t0, 2)

            # ==========================================================
            # 7️⃣ Summary & reporting
            # ==========================================================
            total_time = round(time.time() - start_time, 2)
            self.logger.info(f"[RUN] Pipeline completed successfully in {total_time:.2f}s for split='{self.split}'.")

            summary = {
                "split": self.split,
                "status": "success",
                "duration_total_sec": total_time,
                "phase_durations": phase_times,
                "proportional": prop_report,
                "rebalance": reb_report,
                "focus_zoom": zoom_report,
                "merge": merge_report
            }

            # Save structured report
            try:
                report_path = self.output_dir / f"pipeline_summary_{self.split}.json"
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                self.logger.info(f"[RUN] Full pipeline summary saved to: {report_path}")
            except Exception as e:
                self.logger.warning(f"[RUN] Failed to save summary JSON: {e}")

            return summary

        # ==========================================================
        # 8️⃣ Error handling
        # ==========================================================
        except Exception as e:
            self.logger.error(f"[RUN] Pipeline failed for split='{self.split}': {e}")
            return {
                "split": self.split,
                "status": "failed",
                "error": str(e),
                "duration_total_sec": round(time.time() - start_time, 2)
            }

    def run_stages(self, stages: list):
        """
        Execute specific stages of the pipeline.
        
        Args:
            stages: List of stage names to execute
        """
        available_stages = {
            "prop": self.apply_global_proportional_augmentation,
            "rebalance": lambda: self.apply_rebalance_augmentation() if self.sampling_plan else self.logger.warning("No sampling plan. Run define_rebalance_strategy and plan_rebalance_sampling first."),
            "zoom": self.apply_minority_focus_zoom,
            "merge": self.merge_augmented_splits,
        }
        
        self.logger.info(f"Running specific stages: {stages}")
        
        for stage in stages:
            if stage not in available_stages:
                self.logger.warning(f"Unknown stage: {stage}. Available: {list(available_stages.keys())}")
                continue
                
            self.logger.info(f"Executing stage: {stage}")
            try:
                available_stages[stage]()
                self.logger.info(f"Stage {stage} completed successfully.")
            except Exception as e:
                self.logger.error(f"Stage {stage} failed: {e}")
                raise


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Dataset Augmentation Pipeline")
    parser.add_argument("--config", default="augmentation_config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--split", default="train_joined",
                       help="Dataset split to process")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker threads")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose logging")
    parser.add_argument("--stages", type=str, default="all",
                       help="Comma-separated list of stages to run (prop,rebalance,zoom,merge) or 'all'")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Initialize pipeline
        pipeline = AugmentationPipeline(
            config=config,
            split=args.split,
            logger=None
        )
        
        # Run pipeline
        if args.stages.lower() == "all":
            summary = pipeline.run()
        else:
            stages = [s.strip() for s in args.stages.split(",")]
            pipeline.run_stages(stages)
            summary = {"status": "success", "stages_run": stages}
        
        print(f"\nAugmentation pipeline completed successfully!")
        print(f"Final dataset: {pipeline.final_json}")
        
        return summary
        
    except Exception as e:
        print(f"Augmentation pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
