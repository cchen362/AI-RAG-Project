#!/usr/bin/env python3
"""
Health Check Script for AI-RAG-Project
Verifies that the application is running correctly in Docker
"""

import sys
import os
import json
import requests
import time
from pathlib import Path

def check_streamlit_health():
    """Check if Streamlit app is responding"""
    try:
        # Try Streamlit health endpoint
        response = requests.get('http://localhost:8501/_stcore/health', timeout=5)
        if response.status_code == 200:
            return True, "Streamlit app responding"
        else:
            return False, f"Streamlit returned status {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Streamlit connection failed: {e}"

def check_model_manifest():
    """Check if model manifest exists (indicates successful pre-loading)"""
    manifest_path = Path('/app/models/model_manifest.json')
    
    if not manifest_path.exists():
        return False, "Model manifest not found"
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        loaded_models = manifest.get('total_models', 0)
        if loaded_models > 0:
            return True, f"Model manifest OK ({loaded_models} models loaded)"
        else:
            return False, "No models found in manifest"
            
    except Exception as e:
        return False, f"Failed to read manifest: {e}"

def check_environment():
    """Check essential environment variables"""
    required_env = ['DOCKER_PRELOADED_MODELS', 'TRANSFORMERS_CACHE', 'HF_HOME']
    missing = []
    
    for env_var in required_env:
        if not os.getenv(env_var):
            missing.append(env_var)
    
    if missing:
        return False, f"Missing environment variables: {', '.join(missing)}"
    else:
        return True, "Environment variables OK"

def check_model_cache():
    """Check if model cache directories exist"""
    cache_dirs = [
        '/app/models/transformers',
        '/app/models/huggingface'
    ]
    
    existing = []
    missing = []
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            existing.append(cache_dir)
        else:
            missing.append(cache_dir)
    
    if existing:
        return True, f"Model cache OK ({len(existing)}/{len(cache_dirs)} directories)"
    else:
        return False, f"No model cache directories found: {missing}"

def check_cpu_compatibility():
    """Check CPU compatibility for AI libraries"""
    try:
        # Check if GPU-only mode is enforced
        model_device = os.getenv('MODEL_DEVICE', 'auto')
        if model_device == 'cuda':
            return True, "GPU-only mode enforced - CPU compatibility not required"
        
        # Try to import AI libraries that might fail on old CPUs
        try:
            import sentence_transformers
            import torch
            
            # Test basic tensor operations that require modern instruction sets
            test_tensor = torch.randn(10, 10)
            test_result = torch.matmul(test_tensor, test_tensor.t())
            
            return True, "CPU compatible with AI libraries"
            
        except Exception as e:
            return False, f"CPU compatibility issue: {e}"
            
    except Exception as e:
        return False, f"CPU compatibility check failed: {e}"

def check_gpu_availability():
    """Check GPU availability and CUDA setup"""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            cuda_version = torch.version.cuda
            
            # Test GPU memory allocation
            try:
                test_tensor = torch.randn(10, 10).cuda()
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
                torch.cuda.empty_cache()
                
                return True, f"GPU OK - {gpu_name} (CUDA {cuda_version}, {gpu_count} devices, {memory_allocated:.1f}MB test)"
            except Exception as e:
                return False, f"GPU detected but memory allocation failed: {e}"
        else:
            # Check if this is expected (CPU-only deployment)
            model_device = os.getenv('MODEL_DEVICE', 'cpu')
            if model_device == 'cpu':
                return True, "CPU mode - GPU not required"
            else:
                return False, "GPU expected but not available"
                
    except ImportError:
        return False, "PyTorch not available for GPU check"
    except Exception as e:
        return False, f"GPU check failed: {e}"

def check_docker_optimization():
    """Check Docker-specific optimizations"""
    checks = []
    
    # Check if running in Docker
    if os.path.exists('/.dockerenv'):
        checks.append("Docker environment detected")
    else:
        checks.append("Not running in Docker (local mode)")
    
    # Check pre-loaded models flag
    if os.getenv('DOCKER_PRELOADED_MODELS') == 'true':
        checks.append("Pre-loaded models enabled")
    else:
        checks.append("Pre-loaded models disabled")
    
    # Check model device setting
    model_device = os.getenv('MODEL_DEVICE', 'auto')
    checks.append(f"Model device: {model_device}")
    
    return True, " | ".join(checks)

def main():
    """Run all health checks"""
    print("🏥 AI-RAG-Project Health Check")
    print("=" * 40)
    
    checks = [
        ("Environment", check_environment),
        ("Docker Config", check_docker_optimization),
        ("CPU Compatibility", check_cpu_compatibility),
        ("Model Cache", check_model_cache),
        ("Model Manifest", check_model_manifest),
        ("GPU/CUDA", check_gpu_availability),
        ("Streamlit App", check_streamlit_health)
    ]
    
    all_passed = True
    results = []
    
    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check_name}: {message}")
            results.append(passed)
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"❌ FAIL {check_name}: Exception - {e}")
            results.append(False)
            all_passed = False
    
    print("=" * 40)
    
    if all_passed:
        print("🎉 All health checks passed!")
        print("📊 Application is healthy and ready")
        sys.exit(0)
    else:
        failed_count = len([r for r in results if not r])
        print(f"⚠️ {failed_count}/{len(checks)} health checks failed")
        print("🔧 Application may need attention")
        sys.exit(1)

if __name__ == "__main__":
    main()