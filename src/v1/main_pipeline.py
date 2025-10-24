#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main_pipeline.py
Pipeline principal que ejecuta todo el flujo de procesamiento de datos.

Este script integra y ejecuta secuencialmente:
1. Pipeline de calidad (quality_pipeline.py)
2. Pipeline de aumentación (augment_pipeline.py)
3. Preparación para entrenamiento de modelos

Uso:
    python main_pipeline.py --config quality_config.yaml --augment-config augmentation_config.yaml
    python main_pipeline.py --config quality_config.yaml --augment-config augmentation_config.yaml --stages quality,augment
    python main_pipeline.py --config quality_config.yaml --augment-config augmentation_config.yaml --skip-augment
"""

import argparse
import json
import logging
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Import our pipeline modules
from quality_pipeline import DatasetChecker
from augment_pipeline import AugmentationPipeline


class MainPipeline:
    """
    Pipeline principal que coordina la ejecución de todos los sub-pipelines.
    """
    
    def __init__(self, quality_config_path: str, augment_config_path: Optional[str] = None, 
                 verbose: bool = True, workers: int = 4):
        """
        Initialize the main pipeline with configuration files.
        
        Args:
            quality_config_path: Path to quality pipeline configuration
            augment_config_path: Path to augmentation pipeline configuration (optional)
            verbose: Enable verbose logging
            workers: Number of worker threads
        """
        self.quality_config_path = Path(quality_config_path)
        self.augment_config_path = Path(augment_config_path) if augment_config_path else None
        self.verbose = verbose
        self.workers = workers
        
        # Setup logging
        self._setup_logging()
        
        # Load configurations
        self.quality_config = self._load_config(self.quality_config_path)
        self.augment_config = self._load_config(self.augment_config_path) if self.augment_config_path else None
        
        # Initialize pipeline components
        self.quality_checker = None
        self.augment_pipeline = None
        
        # Results tracking
        self.results = {
            "quality": None,
            "augment": None,
            "total_duration": 0.0,
            "status": "pending"
        }
        
        self.logger.info("MainPipeline initialized successfully")
        self.logger.info(f"Quality config: {self.quality_config_path}")
        if self.augment_config_path:
            self.logger.info(f"Augment config: {self.augment_config_path}")
        else:
            self.logger.info("Augmentation disabled (no config provided)")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO if self.verbose else logging.WARNING
        
        # Create logger
        self.logger = logging.getLogger("MainPipeline")
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file."""
        if not config_path or not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from: {config_path}")
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration {config_path}: {e}")
    
    def run_quality_pipeline(self) -> Dict:
        """
        Execute the quality pipeline (verification, validation, mirror creation).
        
        Returns:
            Dictionary with quality pipeline results
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 1: QUALITY PIPELINE")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Initialize quality checker
            self.quality_checker = DatasetChecker(
                config_path=str(self.quality_config_path),
                num_workers=self.workers,
                verbose=self.verbose
            )
            
            # Run full quality check
            quality_results = self.quality_checker.run_full_check()
            
            duration = time.time() - start_time
            self.logger.info(f"Quality pipeline completed in {duration:.2f} seconds")
            
            self.results["quality"] = {
                "status": "success",
                "duration": duration,
                "results": quality_results
            }
            
            return self.results["quality"]
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Quality pipeline failed: {e}")
            
            self.results["quality"] = {
                "status": "failed",
                "duration": duration,
                "error": str(e)
            }
            
            raise
    
    def run_augmentation_pipeline(self, split: str = "train_joined") -> Dict:
        """
        Execute the augmentation pipeline.
        
        Args:
            split: Dataset split to augment (default: train_joined)
            
        Returns:
            Dictionary with augmentation pipeline results
        """
        if not self.augment_config:
            self.logger.warning("Augmentation config not provided. Skipping augmentation pipeline.")
            return {"status": "skipped", "reason": "no_config"}
        
        self.logger.info("=" * 80)
        self.logger.info("STAGE 2: AUGMENTATION PIPELINE")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Initialize augmentation pipeline
            self.augment_pipeline = AugmentationPipeline(
                config=self.augment_config,
                split=split,
                logger=self.logger
            )
            
            # Run augmentation pipeline
            augment_results = self.augment_pipeline.run()
            
            duration = time.time() - start_time
            self.logger.info(f"Augmentation pipeline completed in {duration:.2f} seconds")
            
            self.results["augment"] = {
                "status": "success",
                "duration": duration,
                "results": augment_results
            }
            
            return self.results["augment"]
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Augmentation pipeline failed: {e}")
            
            self.results["augment"] = {
                "status": "failed",
                "duration": duration,
                "error": str(e)
            }
            
            raise
    
    def prepare_training_data(self) -> Dict:
        """
        Prepare final training data and create training-ready datasets.
        
        Returns:
            Dictionary with preparation results
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 3: TRAINING DATA PREPARATION")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Determine final dataset paths
            if self.augment_pipeline and self.augment_pipeline.final_json.exists():
                # Use augmented dataset
                final_json = self.augment_pipeline.final_json
                final_dir = self.augment_pipeline.final_dir
                self.logger.info(f"Using augmented dataset: {final_json}")
            elif self.quality_checker:
                # Use quality-checked dataset
                mirror_dir = self.quality_checker.output_dir / "mirror_clean"
                final_json = mirror_dir / "train_joined" / "train_joined.json"
                final_dir = mirror_dir / "train_joined"
                self.logger.info(f"Using quality-checked dataset: {final_json}")
            else:
                raise RuntimeError("No processed dataset available")
            
            # Verify final dataset exists
            if not final_json.exists():
                raise FileNotFoundError(f"Final dataset not found: {final_json}")
            
            # Create training-ready directory structure
            training_dir = Path("../../data/training_ready")
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy final dataset to training directory
            training_json = training_dir / "train_final.json"
            training_images = training_dir / "images"
            training_images.mkdir(exist_ok=True)
            
            # Copy JSON
            import shutil
            shutil.copy2(final_json, training_json)
            
            # Copy images
            if final_dir.exists():
                final_images_dir = final_dir / "images"
                if final_images_dir.exists():
                    for img_file in final_images_dir.glob("*"):
                        if img_file.is_file():
                            shutil.copy2(img_file, training_images / img_file.name)
            
            # Create dataset summary
            with open(final_json, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            
            summary = {
                "images": len(dataset.get("images", [])),
                "annotations": len(dataset.get("annotations", [])),
                "categories": len(dataset.get("categories", [])),
                "training_json": str(training_json),
                "training_images": str(training_images)
            }
            
            duration = time.time() - start_time
            self.logger.info(f"Training data preparation completed in {duration:.2f} seconds")
            self.logger.info(f"Final dataset: {summary['images']} images, {summary['annotations']} annotations")
            self.logger.info(f"Training data saved to: {training_dir}")
            
            return {
                "status": "success",
                "duration": duration,
                "summary": summary
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Training data preparation failed: {e}")
            return {
                "status": "failed",
                "duration": duration,
                "error": str(e)
            }
    
    def run_full_pipeline(self, skip_augment: bool = False, augment_split: str = "train_joined") -> Dict:
        """
        Execute the complete pipeline: quality -> augmentation -> training preparation.
        
        Args:
            skip_augment: Skip augmentation pipeline
            augment_split: Dataset split to augment
            
        Returns:
            Dictionary with complete pipeline results
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPLETE DATA PROCESSING PIPELINE")
        self.logger.info("=" * 80)
        
        pipeline_start = time.time()
        
        try:
            # Stage 1: Quality Pipeline
            quality_results = self.run_quality_pipeline()
            if quality_results["status"] != "success":
                raise RuntimeError("Quality pipeline failed")
            
            # Stage 2: Augmentation Pipeline (optional)
            if not skip_augment and self.augment_config:
                augment_results = self.run_augmentation_pipeline(split=augment_split)
                if augment_results["status"] not in ["success", "skipped"]:
                    self.logger.warning("Augmentation pipeline failed, continuing with quality-checked data")
            else:
                self.logger.info("Skipping augmentation pipeline")
                self.results["augment"] = {"status": "skipped", "reason": "disabled"}
            
            # Stage 3: Training Data Preparation
            training_results = self.prepare_training_data()
            if training_results["status"] != "success":
                raise RuntimeError("Training data preparation failed")
            
            # Final summary
            total_duration = time.time() - pipeline_start
            self.results["total_duration"] = total_duration
            self.results["status"] = "success"
            self.results["training"] = training_results
            
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total duration: {total_duration:.2f} seconds")
            self.logger.info(f"Quality: {self.results['quality']['status']}")
            self.logger.info(f"Augment: {self.results['augment']['status']}")
            self.logger.info(f"Training: {self.results['training']['status']}")
            
            # Save results
            self._save_results()
            
            return self.results
            
        except Exception as e:
            total_duration = time.time() - pipeline_start
            self.results["total_duration"] = total_duration
            self.results["status"] = "failed"
            self.results["error"] = str(e)
            
            self.logger.error("=" * 80)
            self.logger.error("PIPELINE FAILED")
            self.logger.error("=" * 80)
            self.logger.error(f"Error: {e}")
            self.logger.error(f"Duration: {total_duration:.2f} seconds")
            
            # Save results even on failure
            self._save_results()
            
            raise
    
    def run_stages(self, stages: List[str], **kwargs) -> Dict:
        """
        Execute specific stages of the pipeline.
        
        Args:
            stages: List of stage names to execute
            **kwargs: Additional arguments for specific stages
            
        Returns:
            Dictionary with results
        """
        self.logger.info(f"Running specific stages: {stages}")
        
        available_stages = {
            "quality": self.run_quality_pipeline,
            "augment": lambda: self.run_augmentation_pipeline(kwargs.get("augment_split", "train_joined")),
            "training": self.prepare_training_data,
        }
        
        results = {}
        
        for stage in stages:
            if stage not in available_stages:
                self.logger.warning(f"Unknown stage: {stage}. Available: {list(available_stages.keys())}")
                continue
            
            self.logger.info(f"Executing stage: {stage}")
            try:
                stage_result = available_stages[stage]()
                results[stage] = stage_result
                self.logger.info(f"Stage {stage} completed successfully.")
            except Exception as e:
                self.logger.error(f"Stage {stage} failed: {e}")
                results[stage] = {"status": "failed", "error": str(e)}
                raise
        
        return results
    
    def _save_results(self):
        """Save pipeline results to JSON file."""
        try:
            results_dir = Path("../../data/outputs")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = results_dir / "main_pipeline_results.json"
            
            # Convert Path objects to strings for JSON serialization
            def convert_paths(obj):
                if isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_paths(self.results)
            
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Pipeline results saved to: {results_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save results: {e}")
    
    def get_training_paths(self) -> Dict[str, str]:
        """
        Get paths to training-ready datasets.
        
        Returns:
            Dictionary with training dataset paths
        """
        if not self.results.get("training"):
            raise RuntimeError("Training data not prepared. Run pipeline first.")
        
        return self.results["training"]["summary"]


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Main Data Processing Pipeline")
    parser.add_argument("--config", required=True,
                       help="Path to quality pipeline configuration file")
    parser.add_argument("--augment-config",
                       help="Path to augmentation pipeline configuration file")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker threads")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose logging")
    parser.add_argument("--skip-augment", action="store_true",
                       help="Skip augmentation pipeline")
    parser.add_argument("--augment-split", default="train_joined",
                       help="Dataset split to augment")
    parser.add_argument("--stages", type=str,
                       help="Comma-separated list of stages to run (quality,augment,training)")
    
    args = parser.parse_args()
    
    try:
        # Initialize main pipeline
        pipeline = MainPipeline(
            quality_config_path=args.config,
            augment_config_path=args.augment_config,
            verbose=args.verbose,
            workers=args.workers
        )
        
        # Run pipeline
        if args.stages:
            # Run specific stages
            stages = [s.strip() for s in args.stages.split(",")]
            results = pipeline.run_stages(
                stages=stages,
                augment_split=args.augment_split
            )
        else:
            # Run full pipeline
            results = pipeline.run_full_pipeline(
                skip_augment=args.skip_augment,
                augment_split=args.augment_split
            )
        
        # Print final summary
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Status: {results['status'].upper()}")
        print(f"Total Duration: {results['total_duration']:.2f} seconds")
        
        if results.get("quality"):
            print(f"Quality Pipeline: {results['quality']['status']}")
        if results.get("augment"):
            print(f"Augmentation Pipeline: {results['augment']['status']}")
        if results.get("training"):
            print(f"Training Preparation: {results['training']['status']}")
        
        print("=" * 80)
        
        if results["status"] == "success":
            print("✅ Pipeline completed successfully!")
            training_paths = pipeline.get_training_paths()
            print(f"📁 Training data ready at: {training_paths['training_json']}")
        else:
            print("❌ Pipeline failed!")
            sys.exit(1)
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
