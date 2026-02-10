"""
Utility functions for file and model management
"""

from pathlib import Path
import os


def check_model_files(models_dir: str) -> dict:
    """
    Check if all required model files exist.
    
    Args:
        models_dir: Path to models directory
        
    Returns:
        Dictionary with model names and availability status
    """
    models_dir = Path(models_dir)
    
    required_models = {
        'hand_landmarker.task': 'Hand Detection Model',
        'face_landmarker.task': 'Face Landmark Model',
        'blaze_face_short_range.tflite': 'Face Detection Model (Optional)',
    }
    
    status = {}
    for model_file, description in required_models.items():
        model_path = models_dir / model_file
        status[model_file] = {
            'description': description,
            'exists': model_path.exists(),
            'path': str(model_path)
        }
    
    return status


def print_model_status(status: dict):
    """
    Print model availability status.
    
    Args:
        status: Status dictionary from check_model_files
    """
    print("\n[INFO] Model Files Status:")
    print("=" * 60)
    
    for model_file, info in status.items():
        exists_str = "✓ Found" if info['exists'] else "✗ Missing"
        print(f"{exists_str:12} | {info['description']:30}")
        if not info['exists']:
            print(f"             | Expected: {info['path']}")
    
    print("=" * 60)


def ensure_directories(project_root: str):
    """
    Ensure all required directories exist.
    
    Args:
        project_root: Project root directory
    """
    directories = [
        'models',
        'sample_data',
        'src',
        'src/detectors',
        'src/utils',
    ]
    
    for directory in directories:
        path = Path(project_root) / directory
        path.mkdir(parents=True, exist_ok=True)
