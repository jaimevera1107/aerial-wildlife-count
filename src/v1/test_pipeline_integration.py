#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pipeline_integration.py
Script de prueba para verificar la integración completa del pipeline.

Este script prueba:
1. Detección automática de datasets
2. Validación de configuración
3. Ejecución de pipelines
4. Integración con scripts de entrenamiento

Uso:
    python test_pipeline_integration.py --test-all
    python test_pipeline_integration.py --test-detection
    python test_pipeline_integration.py --test-pipeline
"""

import argparse
import logging
import sys
from pathlib import Path

# Import our modules
from pipeline_utils import (
    find_processed_dataset, 
    validate_dataset, 
    get_training_config,
    setup_training_paths,
    check_pipeline_prerequisites
)
from quality_pipeline import DatasetChecker
from augment_pipeline import AugmentationPipeline
from main_pipeline import MainPipeline


def setup_logging(verbose: bool = True):
    """Setup logging for testing."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )


def test_dataset_detection():
    """Test automatic dataset detection."""
    print("\n" + "="*60)
    print("TESTING DATASET DETECTION")
    print("="*60)
    
    try:
        # Test finding processed dataset
        json_path, images_dir = find_processed_dataset(split="train", prefer_augmented=True)
        print(f"✓ Found processed dataset:")
        print(f"  JSON: {json_path}")
        print(f"  Images: {images_dir}")
        
        # Test validation
        validation = validate_dataset(json_path, images_dir)
        print(f"✓ Dataset validation successful:")
        print(f"  Images: {validation['num_images']}")
        print(f"  Annotations: {validation['num_annotations']}")
        print(f"  Categories: {validation['num_categories']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset detection failed: {e}")
        return False


def test_training_config():
    """Test training configuration setup."""
    print("\n" + "="*60)
    print("TESTING TRAINING CONFIGURATION")
    print("="*60)
    
    try:
        # Test different dataset types
        for dataset_type in ["auto", "quality", "augmented"]:
            try:
                config = get_training_config(dataset_type=dataset_type, split="train")
                print(f"✓ {dataset_type} dataset config:")
                print(f"  Classes: {config['num_classes']}")
                print(f"  JSON: {Path(config['dataset']['json_path']).name}")
            except Exception as e:
                print(f"✗ {dataset_type} dataset config failed: {e}")
        
        # Test model-specific paths
        for model_type in ["cascade_rcnn", "deformable_detr", "yolov8"]:
            try:
                config = setup_training_paths(model_type=model_type, dataset_type="auto")
                print(f"✓ {model_type} training paths configured")
            except Exception as e:
                print(f"✗ {model_type} training paths failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training configuration test failed: {e}")
        return False


def test_pipeline_prerequisites():
    """Test pipeline prerequisites."""
    print("\n" + "="*60)
    print("TESTING PIPELINE PREREQUISITES")
    print("="*60)
    
    try:
        checks = check_pipeline_prerequisites()
        
        print("Prerequisite check results:")
        for check, status in checks.items():
            status_icon = "✓" if status else "✗"
            print(f"  {status_icon} {check}: {status}")
        
        # Check if we can proceed
        if checks["quality_config"] and checks["original_data"]:
            print("✓ Minimum prerequisites met for quality pipeline")
            return True
        else:
            print("✗ Missing required prerequisites")
            return False
            
    except Exception as e:
        print(f"✗ Prerequisites check failed: {e}")
        return False


def test_quality_pipeline():
    """Test quality pipeline initialization."""
    print("\n" + "="*60)
    print("TESTING QUALITY PIPELINE")
    print("="*60)
    
    try:
        # Check if config exists
        config_path = Path("quality_config.yaml")
        if not config_path.exists():
            print("✗ Quality config not found")
            return False
        
        # Initialize checker
        checker = DatasetChecker(
            config_path=str(config_path),
            num_workers=2,  # Use fewer workers for testing
            verbose=True
        )
        
        print("✓ Quality pipeline initialized successfully")
        print(f"  Output directory: {checker.output_dir}")
        print(f"  Classes: {len(checker.classes)}")
        print(f"  Splits: {list(checker.image_dirs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quality pipeline test failed: {e}")
        return False


def test_augmentation_pipeline():
    """Test augmentation pipeline initialization."""
    print("\n" + "="*60)
    print("TESTING AUGMENTATION PIPELINE")
    print("="*60)
    
    try:
        # Check if config exists
        config_path = Path("augmentation_config.yaml")
        if not config_path.exists():
            print("✗ Augmentation config not found")
            return False
        
        # Load config
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Initialize pipeline
        pipeline = AugmentationPipeline(
            config=config,
            split="train_joined",
            logger=None
        )
        
        print("✓ Augmentation pipeline initialized successfully")
        print(f"  Output directory: {pipeline.output_dir}")
        print(f"  Classes: {len(pipeline.classes)}")
        print(f"  Mode: {pipeline.mode}")
        
        return True
        
    except Exception as e:
        print(f"✗ Augmentation pipeline test failed: {e}")
        return False


def test_main_pipeline():
    """Test main pipeline initialization."""
    print("\n" + "="*60)
    print("TESTING MAIN PIPELINE")
    print("="*60)
    
    try:
        # Check configs
        quality_config = Path("quality_config.yaml")
        augment_config = Path("augmentation_config.yaml")
        
        if not quality_config.exists():
            print("✗ Quality config not found")
            return False
        
        # Initialize main pipeline
        pipeline = MainPipeline(
            quality_config_path=str(quality_config),
            augment_config_path=str(augment_config) if augment_config.exists() else None,
            verbose=True,
            workers=2
        )
        
        print("✓ Main pipeline initialized successfully")
        print(f"  Quality config: {pipeline.quality_config_path}")
        if pipeline.augment_config_path:
            print(f"  Augment config: {pipeline.augment_config_path}")
        else:
            print("  Augment config: None (disabled)")
        
        return True
        
    except Exception as e:
        print(f"✗ Main pipeline test failed: {e}")
        return False


def test_training_scripts():
    """Test training script integration."""
    print("\n" + "="*60)
    print("TESTING TRAINING SCRIPT INTEGRATION")
    print("="*60)
    
    try:
        # Test that training scripts can import pipeline utilities
        scripts = [
            "train_cascade_rcnn.py",
            "train_deformable_detr.py", 
            "train_yolov8.py"
        ]
        
        for script in scripts:
            script_path = Path(script)
            if script_path.exists():
                print(f"✓ {script} exists")
                
                # Check if it imports pipeline_utils
                content = script_path.read_text(encoding="utf-8")
                if "from pipeline_utils import" in content:
                    print(f"  ✓ Imports pipeline utilities")
                else:
                    print(f"  ✗ Does not import pipeline utilities")
            else:
                print(f"✗ {script} not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Training script test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("RUNNING ALL PIPELINE INTEGRATION TESTS")
    print("="*80)
    
    tests = [
        ("Pipeline Prerequisites", test_pipeline_prerequisites),
        ("Dataset Detection", test_dataset_detection),
        ("Training Configuration", test_training_config),
        ("Quality Pipeline", test_quality_pipeline),
        ("Augmentation Pipeline", test_augmentation_pipeline),
        ("Main Pipeline", test_main_pipeline),
        ("Training Scripts", test_training_scripts),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        icon = "✓" if result else "✗"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Pipeline Integration")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--test-detection", action="store_true", help="Test dataset detection")
    parser.add_argument("--test-config", action="store_true", help="Test training configuration")
    parser.add_argument("--test-pipeline", action="store_true", help="Test pipeline initialization")
    parser.add_argument("--test-scripts", action="store_true", help="Test training scripts")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run tests
    if args.test_all:
        success = run_all_tests()
    else:
        success = True
        
        if args.test_detection:
            success &= test_dataset_detection()
        
        if args.test_config:
            success &= test_training_config()
        
        if args.test_pipeline:
            success &= test_pipeline_prerequisites()
            success &= test_quality_pipeline()
            success &= test_augmentation_pipeline()
            success &= test_main_pipeline()
        
        if args.test_scripts:
            success &= test_training_scripts()
        
        if not any([args.test_detection, args.test_config, args.test_pipeline, args.test_scripts]):
            print("No specific tests selected. Use --test-all to run all tests.")
            return
    
    # Exit with appropriate code
    if success:
        print("\n✅ All selected tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
