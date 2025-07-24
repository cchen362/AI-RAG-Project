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

def main():
    """Run all health checks"""
    print("🏥 AI-RAG-Project Health Check")
    print("=" * 40)
    
    checks = [
        ("Environment", check_environment),
        ("Model Cache", check_model_cache),
        ("Model Manifest", check_model_manifest),
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