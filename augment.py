# Versión ajustada del script de augmentación / AugmentationPipeline
# Notas: mantiene nombres de clase/métodos/variables originales.

import os
import json
import logging
import threading
import itertools
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image, ImageOps
import numpy as np
import hashlib
import traceback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# try importing albumentations; if not installed, raise informative error
try:
    import albumentations as A
except Exception:
    A = None
    logger.warning("Albumentations not available; prepare_transforms will fail if called.")

# reuse the same canonical hashing helper for consistency
def image_content_hash(path_or_image, resize_to: None = None) -> str:
    # Implementation identical to the function used in quality_fixed.py
    try:
        if isinstance(path_or_image, (str, Path)):
            p = Path(path_or_image)
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
        elif isinstance(path_or_image, Image.Image):
            img = path_or_image
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        else:
            raise ValueError("Unsupported type for image_content_hash")

        if resize_to is not None:
            img = img.resize(resize_to, Image.Resampling.LANCZOS)

        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        m = hashlib.md5()
        m.update(str(arr.shape).encode("utf-8"))
        m.update(arr.tobytes())
        return m.hexdigest()
    except Exception:
        logger.exception("Failed to compute image_content_hash for %s", path_or_image)
        try:
            p = Path(path_or_image)
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            logger.exception("Fallback file MD5 also failed for %s", path_or_image)
            return ""

class AugmentationPipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        config: configuration dict (preserves original structure/keys expected by script)
        Example keys expected:
            - out_dir
            - seed
            - augmentations: dict with 'transforms' list
            - proportion
            - max_repeats
            - rebalance settings...
            - clean_json path or dict
        """
        self.config = config
        self.out_dir = Path(config.get("out_dir", "out_aug"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(config.get("seed", 42))
        self._set_seed_everywhere(self.seed)
        self.pipeline = None
        self.prepare_transforms()
        # counters for id allocation (kept same names)
        self.img_id_counter = itertools.count(int(config.get("start_img_id", 1000000)))
        self.ann_id_counter = itertools.count(int(config.get("start_ann_id", 1000000)))
        self.id_lock = threading.Lock()
        # load input JSON if provided (clean_json may be path or dict)
        self.clean_json = None
        cj = config.get("clean_json")
        if isinstance(cj, (str, Path)):
            try:
                with open(cj, "r", encoding="utf-8") as f:
                    self.clean_json = json.load(f)
            except Exception:
                logger.exception("Failed loading clean_json from %s", cj)
                self.clean_json = None
        else:
            self.clean_json = cj

    def _set_seed_everywhere(self, seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    def prepare_transforms(self):
        """
        Construct albumentations ReplayCompose pipeline with bbox params.
        This method keeps the original idea (OneOf/SomeOf) but ensures correct syntax
        and robust fallback behavior.
        """
        if A is None:
            logger.error("Albumentations not installed; prepare_transforms cannot build pipeline.")
            self.pipeline = None
            return

        aug_blocks = []
        transforms_cfg = self.config.get("augmentations", {}).get("transforms", [])
        valid_params = {"p", "limit", "interpolation", "always_apply"}
        for t in transforms_cfg:
            # expected t is dict with 'name' and 'params'
            name = t.get("name")
            params = t.get("params", {})
            params = {k: v for k, v in params.items() if k in valid_params or k == 'value' or k == 'angle' or k == 'scale'}
            try:
                aug_cls = getattr(A, name)
            except Exception:
                logger.exception("Unknown albumentations transform %s", name)
                continue
            try:
                aug_blocks.append(aug_cls(**params))
            except Exception:
                logger.exception("Failed to instantiate transform %s with params %s", name, params)
                continue

        if not aug_blocks:
            # fallback to some safe basic transforms
            aug_blocks = [A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                          A.RandomBrightnessContrast(p=0.5)]

        # Use OneOf to randomly select one augmentation from the list when executed
        one_of = A.OneOf(aug_blocks, p=1.0)

        # Compose with bbox params. min_visibility taken from config or default 0.25
        min_visibility = float(self.config.get("bbox_min_visibility", 0.25))
        bbox_params = A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=min_visibility)
        # ReplayCompose to allow replaying same transformation if needed downstream
        self.pipeline = A.ReplayCompose([one_of], bbox_params=bbox_params, p=1.0)
        logger.info("Prepared albumentations pipeline with %d blocks", len(aug_blocks))

    def apply_global_proportional_augmentation(self):
        """
        Apply augmentation proportionally to the training set only.
        Safeguard: if entries in self.clean_json contain 'split' field, only images with split == 'train'
        will be considered for augmentation. If no split info available, behavior falls back to original:
        sample from full set.
        """
        if self.pipeline is None:
            logger.error("Pipeline not prepared; aborting augmentation.")
            return None

        if not self.clean_json:
            logger.error("No clean_json loaded; aborting augmentation.")
            return None

        images = self.clean_json.get("images", [])
        annotations = self.clean_json.get("annotations", [])
        anns_by_image = {}
        for a in annotations:
            anns_by_image.setdefault(a["image_id"], []).append(a)

        # decide candidate images: prefer those explicitly marked as train
        images_with_split = any("split" in im for im in images)
        if images_with_split:
            candidates = [im for im in images if im.get("split") == "train"]
            logger.info("Detected 'split' metadata; restricting augmentation to %d train images", len(candidates))
        else:
            # legacy behavior: sample from full images list
            candidates = images
            logger.info("No 'split' metadata detected; augmentation sampling from all images (%d)", len(candidates))

        total = len(candidates)
        proportion = float(self.config.get("proportion", 0.10))
        n_to_aug = max(1, int(total * proportion))
        # use numpy generator for reproducibility
        rng = np.random.default_rng(self.seed)
        idxs = rng.choice(len(candidates), size=n_to_aug, replace=False)
        selected = [candidates[i] for i in idxs]
        out_images = list(images)  # start with originals
        out_annotations = list(annotations)

        logger.info("Applying augmentation to %d images (proportion %.3f)", n_to_aug, proportion)

        for src_im in selected:
            src_path = src_im.get("file_name")
            if not src_path or not os.path.exists(src_path):
                logger.warning("Source image missing, skipping: %s", src_path)
                continue
            src_ann_list = anns_by_image.get(src_im["id"], [])
            try:
                with Image.open(src_path) as pil_img:
                    pil_img = ImageOps.exif_transpose(pil_img)
                    pil_img = pil_img.convert("RGB")
                    arr = np.asarray(pil_img)

                    # build albumentations input: bboxes in COCO format [x,y,w,h]
                    bboxes = []
                    category_ids = []
                    for a in src_ann_list:
                        bboxes.append(a["bbox"])
                        category_ids.append(a["category_id"])

                    # apply augmentations up to max_repeats times but ensure diversity:
                    max_repeats = int(self.config.get("max_repeats", 1))
                    # conservative limit: don't exceed 15 as a safety cap unless explicitly configured higher
                    safety_cap = int(self.config.get("safety_cap_repeats", 15))
                    max_repeats = min(max_repeats, safety_cap)

                    repeated = 0
                    trials = 0
                    target_repeats = max_repeats
                    while repeated < target_repeats and trials < target_repeats * 4:
                        trials += 1
                        try:
                            augmented = self.pipeline(image=arr, bboxes=bboxes, category_ids=category_ids)
                        except Exception:
                            logger.exception("Augmentation pipeline failed on %s", src_path)
                            break

                        aug_image = augmented["image"]
                        aug_bboxes = augmented.get("bboxes", [])
                        aug_category_ids = augmented.get("category_ids", [])

                        # basic validity filter: keep only bboxes with positive area and min visibility satisfied by pipeline
                        valid_boxes = []
                        valid_cat = []
                        for bb, cid in zip(aug_bboxes, aug_category_ids):
                            x, y, w, h = bb
                            if w <= 0 or h <= 0:
                                continue
                            valid_boxes.append([float(x), float(y), float(w), float(h)])
                            valid_cat.append(int(cid))

                        if not valid_boxes:
                            # skip this augmentation if no valid boxes remain
                            continue

                        # create new image id and annotation ids thread-safely
                        with self.id_lock:
                            new_img_id = next(self.img_id_counter)
                        new_filename = f"aug_{new_img_id}_{Path(src_path).name}"
                        out_path = self.out_dir / "images" / new_filename
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            Image.fromarray(aug_image).save(out_path, format="JPEG", quality=95)
                        except Exception:
                            logger.exception("Failed to save augmented image %s", out_path)
                            continue

                        # prepare image entry (preserving structure)
                        width, height = Image.fromarray(aug_image).size
                        image_entry = {
                            "id": new_img_id,
                            "file_name": str(out_path),
                            "width": width,
                            "height": height,
                            # maintain provenance
                            "source_image_id": src_im["id"],
                            "source_file": src_path,
                            "source_stage": "proportional"
                        }
                        out_images.append(image_entry)

                        # add annotations with new ids
                        for bb, cid in zip(valid_boxes, valid_cat):
                            with self.id_lock:
                                new_ann_id = next(self.ann_id_counter)
                            ann_entry = {
                                "id": new_ann_id,
                                "image_id": new_img_id,
                                "category_id": cid,
                                "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                                "area": float(bb[2] * bb[3]),
                                "iscrowd": 0,
                                "source_ann": True
                            }
                            out_annotations.append(ann_entry)

                        repeated += 1
                    # end repeats for this image
            except Exception:
                logger.exception("Failed processing source image for augmentation: %s", src_path)
                continue

        # produce final JSON structure matching COCO-like structure
        final_json = {
            "images": out_images,
            "annotations": out_annotations,
            "categories": self.clean_json.get("categories", [])
        }

        # save final JSON
        final_json_path = self.out_dir / "final_augmented.json"
        try:
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(final_json, f, indent=2, ensure_ascii=False)
            logger.info("Saved augmented json to %s", final_json_path)
        except Exception:
            logger.exception("Failed saving final augmented json to %s", final_json_path)

        # optionally run final verification using consistent content hash
        if self.config.get("verify_final", True):
            self._verify_final_split_contents(final_json)

        return final_json

    def _verify_final_split_contents(self, final_json: Dict[str, Any]):
        """
        Verify final augmented images for duplicates and corruption using canonical content hash
        (consistent with DatasetChecker).
        """
        images = final_json.get("images", [])
        duplicate_map = {}
        seen = {}
        corrupted = []
        for im in images:
            path = im.get("file_name")
            if not path or not os.path.exists(path):
                corrupted.append(path)
                continue
            try:
                h = image_content_hash(path)
                if h in seen:
                    duplicate_map.setdefault(h, []).append(path)
                    duplicate_map[h].append(seen[h])
                else:
                    seen[h] = path
            except Exception:
                logger.exception("Error hashing final image %s", path)
                corrupted.append(path)

        report = {
            "num_images": len(images),
            "num_duplicates": len(duplicate_map),
            "duplicates": {k: v[:10] for k, v in duplicate_map.items()},
            "num_corrupted": len(corrupted),
            "corrupted_examples": corrupted[:10]
        }
        report_path = self.out_dir / "final_verification_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info("Saved final verification report to %s", report_path)
        except Exception:
            logger.exception("Failed to save final verification report to %s", report_path)
        return report

    # Optional: compute a basic measure of augmentation diversity (cosmetic)
    def compute_augmentation_diversity(self, final_json_path: str, sample_limit: int = 200):
        """
        Compute a crude diversity proxy using content hashes: ratio of unique hashes / total.
        Saves a small report. This is a fast, file-based diversity proxy that helps detect excessive repetition.
        """
        try:
            with open(final_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.exception("Failed to load final_json for diversity")
            return None

        images = data.get("images", [])[:sample_limit]
        hashes = []
        for im in images:
            path = im.get("file_name")
            if not path or not os.path.exists(path):
                continue
            h = image_content_hash(path)
            if h:
                hashes.append(h)
        unique = len(set(hashes))
        total = len(hashes)
        ratio = unique / total if total > 0 else 0.0
        report = {"sample_count": total, "unique": unique, "unique_ratio": ratio}
        outp = self.out_dir / "diversity_report.json"
        try:
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to write diversity report")
        return report

