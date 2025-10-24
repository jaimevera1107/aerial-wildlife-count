#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pipeline_utils.py
Utilidades para integrar los pipelines de procesamiento con los scripts de entrenamiento.

Este módulo proporciona funciones auxiliares para:
- Detectar automáticamente datasets procesados
- Configurar rutas de entrenamiento
- Validar datasets antes del entrenamiento
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_processed_dataset(split: str = "train", prefer_augmented: bool = True) -> Tuple[Path, Path]:
    """
    Find the best available processed dataset for training.
    
    Args:
        split: Dataset split to find (train, val, test)
        prefer_augmented: Prefer augmented dataset over quality-checked dataset
        
    Returns:
        Tuple of (json_path, images_dir)
        
    Raises:
        FileNotFoundError: If no suitable dataset is found
    """
    logger = logging.getLogger(__name__)
    
    # Define search paths in order of preference
    search_paths = []
    
    if prefer_augmented:
        # 1. Augmented final dataset
        search_paths.append(("augmented_final", Path("../../data/outputs/mirror_clean") / f"{split}_final" / f"{split}_final.json"))
        # 2. Augmented rebalanced dataset
        search_paths.append(("augmented_rebalance", Path("../../data/outputs/mirror_clean") / f"{split}_rebalance_1" / f"{split}_rebalance_1.json"))
        # 3. Augmented proportional dataset
        search_paths.append(("augmented_prop", Path("../../data/outputs/mirror_clean") / f"{split}_prop" / f"{split}_prop.json"))
    
    # 4. Quality-checked joined dataset
    search_paths.append(("quality_joined", Path("../../data/outputs/mirror_clean") / "train_joined" / "train_joined.json"))
    
    # 5. Quality-checked individual split
    search_paths.append(("quality_split", Path("../../data/outputs/mirror_clean") / f"{split}_clean" / f"{split}_clean.json"))
    
    # 6. Training-ready dataset
    search_paths.append(("training_ready", Path("../../data/training_ready") / "train_final.json"))
    
    # Search for available datasets
    for dataset_type, json_path in search_paths:
        if json_path.exists():
            # Determine images directory
            if dataset_type.startswith("augmented_"):
                images_dir = json_path.parent / "images"
            elif dataset_type == "quality_joined":
                images_dir = json_path.parent / "images"
            elif dataset_type == "quality_split":
                images_dir = json_path.parent / "images"
            elif dataset_type == "training_ready":
                images_dir = json_path.parent / "images"
            else:
                # Fallback: try to find images in parent directory
                images_dir = json_path.parent / "images"
            
            if images_dir.exists():
                logger.info(f"Found {dataset_type} dataset: {json_path}")
                logger.info(f"Images directory: {images_dir}")
                return json_path, images_dir
    
    # If no processed dataset found, try original data
    logger.warning("No processed dataset found. Trying original data...")
    
    original_paths = [
        Path("../../data/groundtruth/json/big_size") / f"{split}_big_size_A_B_E_K_WH_WB.json",
        Path("../../data/groundtruth/json/sub_frames") / f"{split}_subframes_A_B_E_K_WH_WB.json"
    ]
    
    for json_path in original_paths:
        if json_path.exists():
            # Find corresponding images directory
            if "big_size" in str(json_path):
                images_dir = Path("../../data") / split
            else:
                images_dir = Path("../../data") / f"{split}_subframes"
            
            if images_dir.exists():
                logger.info(f"Using original dataset: {json_path}")
                logger.info(f"Images directory: {images_dir}")
                return json_path, images_dir
    
    raise FileNotFoundError(f"No suitable dataset found for split '{split}'")


def validate_dataset(json_path: Path, images_dir: Path, min_images: int = 1, min_annotations: int = 1) -> Dict:
    """
    Validate a dataset before training.
    
    Args:
        json_path: Path to COCO JSON file
        images_dir: Path to images directory
        min_images: Minimum number of images required
        min_annotations: Minimum number of annotations required
        
    Returns:
        Dictionary with validation results
        
    Raises:
        ValueError: If dataset validation fails
    """
    logger = logging.getLogger(__name__)
    
    # Load JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load JSON file {json_path}: {e}")
    
    # Check structure
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in JSON file")
    
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]
    
    # Check counts
    if len(images) < min_images:
        raise ValueError(f"Insufficient images: {len(images)} < {min_images}")
    
    if len(annotations) < min_annotations:
        raise ValueError(f"Insufficient annotations: {len(annotations)} < {min_annotations}")
    
    if len(categories) == 0:
        raise ValueError("No categories defined")
    
    # Check image files exist
    missing_images = []
    for img in images:
        img_path = images_dir / img["file_name"]
        if not img_path.exists():
            missing_images.append(img["file_name"])
    
    if missing_images:
        logger.warning(f"Missing {len(missing_images)} image files (first 5): {missing_images[:5]}")
    
    # Check annotation consistency
    image_ids = {img["id"] for img in images}
    ann_image_ids = {ann["image_id"] for ann in annotations}
    orphan_annotations = ann_image_ids - image_ids
    
    if orphan_annotations:
        logger.warning(f"Found {len(orphan_annotations)} orphan annotations")
    
    # Calculate class distribution
    class_counts = {}
    for ann in annotations:
        cat_id = ann["category_id"]
        class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
    
    # Get class names
    id_to_name = {cat["id"]: cat["name"] for cat in categories}
    class_distribution = {id_to_name.get(cid, f"unknown_{cid}"): count for cid, count in class_counts.items()}
    
    validation_result = {
        "json_path": str(json_path),
        "images_dir": str(images_dir),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "missing_images": len(missing_images),
        "orphan_annotations": len(orphan_annotations),
        "class_distribution": class_distribution,
        "valid": True
    }
    
    logger.info(f"Dataset validation successful:")
    logger.info(f"  Images: {len(images)}")
    logger.info(f"  Annotations: {len(annotations)}")
    logger.info(f"  Categories: {len(categories)}")
    logger.info(f"  Missing images: {len(missing_images)}")
    logger.info(f"  Orphan annotations: {len(orphan_annotations)}")
    
    return validation_result


def get_training_config(dataset_type: str = "auto", split: str = "train") -> Dict:
    """
    Get training configuration based on available dataset.
    
    Args:
        dataset_type: Type of dataset (auto, original, quality, augmented)
        split: Dataset split
        
    Returns:
        Dictionary with training configuration
    """
    logger = logging.getLogger(__name__)
    
    try:
        if dataset_type == "auto":
            json_path, images_dir = find_processed_dataset(split=split)
        else:
            # Use specific dataset type
            if dataset_type == "augmented":
                json_path, images_dir = find_processed_dataset(split=split, prefer_augmented=True)
            elif dataset_type == "quality":
                json_path, images_dir = find_processed_dataset(split=split, prefer_augmented=False)
            else:
                # Original dataset
                json_path, images_dir = find_processed_dataset(split=split, prefer_augmented=False)
        
        # Validate dataset
        validation = validate_dataset(json_path, images_dir)
        
        # Create training config
        config = {
            "dataset": {
                "json_path": str(json_path),
                "images_dir": str(images_dir),
                "validation": validation
            },
            "classes": list(validation["class_distribution"].keys()),
            "num_classes": len(validation["class_distribution"]),
            "dataset_type": dataset_type,
            "split": split
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get training config: {e}")
        raise


def setup_training_paths(model_type: str, dataset_type: str = "auto", split: str = "train") -> Dict:
    """
    Setup training paths for different model types.
    
    Args:
        model_type: Type of model (cascade_rcnn, deformable_detr, yolov8)
        dataset_type: Type of dataset to use
        split: Dataset split
        
    Returns:
        Dictionary with training paths and configuration
    """
    logger = logging.getLogger(__name__)
    
    # Get dataset configuration
    dataset_config = get_training_config(dataset_type=dataset_type, split=split)
    
    # Setup model-specific paths
    if model_type == "cascade_rcnn":
        work_dir = Path("../../work_dirs/cascade")
        work_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "dataset": dataset_config["dataset"],
            "work_dir": str(work_dir),
            "model_type": "cascade_rcnn",
            "classes": dataset_config["classes"],
            "num_classes": dataset_config["num_classes"]
        }
        
    elif model_type == "deformable_detr":
        work_dir = Path("../../work_dirs/deformable_detr")
        work_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "dataset": dataset_config["dataset"],
            "work_dir": str(work_dir),
            "model_type": "deformable_detr",
            "classes": dataset_config["classes"],
            "num_classes": dataset_config["num_classes"]
        }
        
    elif model_type == "yolov8":
        project_dir = Path("../../runs_yolo")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "dataset": dataset_config["dataset"],
            "project": str(project_dir),
            "model_type": "yolov8",
            "classes": dataset_config["classes"],
            "num_classes": dataset_config["num_classes"]
        }
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Training paths configured for {model_type}:")
    logger.info(f"  Dataset: {config['dataset']['json_path']}")
    logger.info(f"  Images: {config['dataset']['images_dir']}")
    logger.info(f"  Classes: {config['num_classes']}")
    
    return config


def check_pipeline_prerequisites() -> Dict:
    """
    Check if pipeline prerequisites are met.
    
    Returns:
        Dictionary with prerequisite check results
    """
    logger = logging.getLogger(__name__)
    
    checks = {
        "quality_config": False,
        "augment_config": False,
        "original_data": False,
        "processed_data": False
    }
    
    # Check configuration files
    quality_config = Path("quality_config.yaml")
    if quality_config.exists():
        checks["quality_config"] = True
        logger.info("✓ Quality configuration found")
    else:
        logger.warning("✗ Quality configuration not found")
    
    augment_config = Path("augmentation_config.yaml")
    if augment_config.exists():
        checks["augment_config"] = True
        logger.info("✓ Augmentation configuration found")
    else:
        logger.warning("✗ Augmentation configuration not found")
    
    # Check original data
    original_data_dirs = [
        Path("../../data/train"),
        Path("../../data/val"),
        Path("../../data/test")
    ]
    
    if all(d.exists() for d in original_data_dirs):
        checks["original_data"] = True
        logger.info("✓ Original data directories found")
    else:
        logger.warning("✗ Some original data directories missing")
    
    # Check processed data
    processed_data_dirs = [
        Path("../../data/outputs/mirror_clean"),
        Path("../../data/training_ready")
    ]
    
    if any(d.exists() for d in processed_data_dirs):
        checks["processed_data"] = True
        logger.info("✓ Processed data found")
    else:
        logger.warning("✗ No processed data found")
    
    return checks


def run_pipeline_if_needed(force: bool = False) -> bool:
    """
    Run the data processing pipeline if needed.
    
    Args:
        force: Force pipeline execution even if processed data exists
        
    Returns:
        True if pipeline was run, False if skipped
    """
    logger = logging.getLogger(__name__)
    
    # Check if processed data already exists
    if not force:
        try:
            find_processed_dataset()
            logger.info("Processed dataset already exists. Skipping pipeline.")
            return False
        except FileNotFoundError:
            logger.info("No processed dataset found. Running pipeline...")
    
    # Check prerequisites
    prerequisites = check_pipeline_prerequisites()
    if not prerequisites["quality_config"]:
        raise RuntimeError("Quality configuration not found. Cannot run pipeline.")
    
    # Import and run main pipeline
    try:
        from main_pipeline import MainPipeline
        
        pipeline = MainPipeline(
            quality_config_path="quality_config.yaml",
            augment_config_path="augmentation_config.yaml" if prerequisites["augment_config"] else None,
            verbose=True
        )
        
        results = pipeline.run_full_pipeline()
        
        if results["status"] == "success":
            logger.info("✓ Pipeline completed successfully")
            return True
        else:
            raise RuntimeError(f"Pipeline failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise
