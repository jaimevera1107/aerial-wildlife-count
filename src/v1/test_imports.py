#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_imports.py
Script simple para verificar que todas las importaciones funcionan correctamente.
"""

def test_imports():
    """Test all module imports."""
    print("Testing module imports...")
    
    try:
        import quality_pipeline
        print("✓ quality_pipeline imported successfully")
    except Exception as e:
        print(f"✗ quality_pipeline import failed: {e}")
        return False
    
    try:
        import augment_pipeline
        print("✓ augment_pipeline imported successfully")
    except Exception as e:
        print(f"✗ augment_pipeline import failed: {e}")
        return False
    
    try:
        import main_pipeline
        print("✓ main_pipeline imported successfully")
    except Exception as e:
        print(f"✗ main_pipeline import failed: {e}")
        return False
    
    try:
        import pipeline_utils
        print("✓ pipeline_utils imported successfully")
    except Exception as e:
        print(f"✗ pipeline_utils import failed: {e}")
        return False
    
    try:
        import test_pipeline_integration
        print("✓ test_pipeline_integration imported successfully")
    except Exception as e:
        print(f"✗ test_pipeline_integration import failed: {e}")
        return False
    
    print("\nAll imports successful! ✓")
    return True

if __name__ == "__main__":
    test_imports()
