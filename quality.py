# Versión ajustada del script de verificación / DatasetChecker
# Notas: mantiene nombres de clase/métodos/variables originales.

import os
import io
import json
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image, ImageOps
import numpy as np
import traceback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --------------------------------------------------------------------
# Helper: canonical content-based image hash (shared by DatasetChecker & AugmentationPipeline)
# --------------------------------------------------------------------
def image_content_hash(path_or_image, resize_to: Optional[tuple] = None) -> str:
    """
    Compute an MD5 hash deterministically over canonical image content.
    Accepts either a file path (str/Path) or a PIL.Image.Image.
    Steps:
      - If path: open with PIL, apply EXIF transpose.
      - Convert to RGB.
      - Optionally resize (maintain aspect ratio if desired; here we use exact resize if provided).
      - Convert to bytes via numpy array.tobytes() and include shape in hash to avoid collisions.
    Returns hex digest string.
    """
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
        # normalize dtype for consistency
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        m = hashlib.md5()
        # include shape so that different sizes with same bytes unlikely collide
        m.update(str(arr.shape).encode("utf-8"))
        m.update(arr.tobytes())
        return m.hexdigest()
    except Exception:
        # Log full stacktrace to help debugging while preserving error surface
        logger.exception("Failed to compute image_content_hash for %s", path_or_image)
        # fallback to file MD5 when reading fails (best-effort)
        try:
            p = Path(path_or_image)
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            logger.exception("Fallback file MD5 also failed for %s", path_or_image)
            return ""

# --------------------------------------------------------------------
# DatasetChecker class 
# --------------------------------------------------------------------
class DatasetChecker:
    def __init__(self, config_path: str):
        """
        Initialization: load configuration, prepare directories, seeds, etc.
        Ensures deterministic seed setup and logger presence.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.out_dir = Path(self.config.get("out_dir", "out"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.out_dir / "cache.json"
        self.history_path = self.out_dir / "history.json"
        self.cache = {}
        self.history = []
        # initialize concurrency helpers used elsewhere
        self._io_semaphore = threading.BoundedSemaphore(value=self.config.get("io_workers", 8))
        # set seed reproducibility
        seed = int(self.config.get("seed", 42))
        self.seed = seed
        self._set_seed(seed)
        logger.info("DatasetChecker initialized with seed %s", seed)
        # load existing cache/history if present
        self._init_cache_and_history()

    def _set_seed(self, seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # try to set deterministic algorithms if available
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # older torch versions
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            # torch not installed: skip
            pass

    def _load_config(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file {path} not found")
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
        # try JSON first, then YAML if available
        try:
            return json.loads(raw)
        except Exception:
            try:
                import yaml
                return yaml.safe_load(raw)
            except Exception:
                logger.exception("Failed to parse config file")
                raise

    def _init_cache_and_history(self):
        # load or initialize cache/history
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                logger.exception("Failed loading cache, starting fresh")
                self.cache = {}
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                logger.exception("Failed loading history, starting fresh")
                self.history = []

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to save cache to %s", self.cache_path)

    def _save_history(self):
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to save history to %s", self.history_path)

    def _log_history(self, entry: Dict[str, Any]):
        self.history.append(entry)
        # keep history size bounded
        if len(self.history) > 5000:
            self.history = self.history[-5000:]
        self._save_history()

    # ---- image processing / verification methods ----
    def verify_images(self, image_paths: List[str], split_name: str = "train"):
        """
        Verify, canonicalize and compute content-hash for a list of image file paths.
        Stores results in self.cache under keys by path.
        Ensures hashing is performed using image_content_hash (content-based) for robust detection.
        """
        results = []
        # thread worker to process single image
        def worker(img_path):
            entry = {"path": str(img_path), "split": split_name}
            try:
                with self._io_semaphore:
                    # canonicalize and compute content hash
                    h = image_content_hash(img_path)
                    entry["hash"] = h
                    # basic checks: try to open and extract shape
                    with Image.open(img_path) as im:
                        im = ImageOps.exif_transpose(im)
                        im = im.convert("RGB")
                        arr = np.asarray(im)
                        entry["width"], entry["height"] = im.size
                        entry["mode"] = im.mode
                        entry["std"] = float(np.std(arr))
                        entry["mean"] = float(np.mean(arr))
                        # mark as valid unless issues found
                        entry["valid"] = True
                        # optionally resave standardized copy if requested
                        if self.config.get("resave_verified", False):
                            target_dir = self.out_dir / "verified_images" / split_name
                            target_dir.mkdir(parents=True, exist_ok=True)
                            target_path = target_dir / Path(img_path).name
                            try:
                                im.save(target_path, format="JPEG", quality=95)
                                entry["verified_path"] = str(target_path)
                            except Exception:
                                logger.exception("Failed to resave verified image %s", img_path)
                # update cache
            except Exception:
                entry["valid"] = False
                entry["error"] = traceback.format_exc()
                logger.exception("Image verification failed for %s", img_path)
            self.cache[str(img_path)] = entry
            return entry

        # execute with ThreadPoolExecutor
        max_workers = int(self.config.get("io_workers", 8))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, p) for p in image_paths]
            for f in as_completed(futures):
                try:
                    res = f.result()
                    results.append(res)
                except Exception:
                    logger.exception("Worker failed in verify_images")
        # persist cache and return summary
        self._save_cache()
        return results

    def _compute_leakage_report(self):
        """
        Compute leakage between splits using canonical content hashes.
        Expects self.cache to contain entries populated by verify_images with 'split' and 'hash'.
        Produces a leakage report dictionary.
        """
        split_hashes = {}
        for p, entry in self.cache.items():
            split = entry.get("split", "train")
            h = entry.get("hash", "")
            if not h:
                continue
            split_hashes.setdefault(split, set()).add(h)

        splits = sorted(split_hashes.keys())
        report = {"intersections": {}, "split_sizes": {s: len(split_hashes[s]) for s in splits}}
        for i in range(len(splits)):
            for j in range(i+1, len(splits)):
                a = splits[i]
                b = splits[j]
                inter = split_hashes[a].intersection(split_hashes[b])
                k = f"{a}__{b}"
                report["intersections"][k] = {"count": len(inter), "examples": list(inter)[:10]}
                if len(inter) > 0:
                    logger.warning("Detected %d overlapping images between %s and %s", len(inter), a, b)

        # save report
        try:
            with open(self.out_dir / "leakage_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to save leakage_report.json")
        return report

    # Example wrapper call to run the main checks (keeps name style)
    def run_full_check(self, images_by_split: Dict[str, List[str]]):
        """
        Run the full verification for supplied dict {split: [image_paths]}.
        This is a convenience orchestrator using verify_images and leakage detection.
        """
        for split, paths in images_by_split.items():
            logger.info("Verifying %d images for split %s", len(paths), split)
            self.verify_images(paths, split_name=split)
        report = self._compute_leakage_report()
        # save cache to disk
        self._save_cache()
        return report

# The module can be extended with other methods (annotation validation etc.)
# but the critical change here is the use of image_content_hash consistently,
# and better exception logging to avoid silent failures.

