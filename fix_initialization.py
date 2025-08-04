#!/usr/bin/env python3
"""
Emergency Fix Script for AI-RAG System Initialization Issues
Run this script to diagnose and fix common initialization problems
"""

import os
import sys
import requests
import time
import json
from pathlib import Path

def check_streamlit_health():
    """Check if Streamlit is running and responsive"""
    try:
        response = requests.get('http://localhost:8502/_stcore/health', timeout=5)
        return response.status_code == 200 and response.text.strip() == 'ok'
    except:
        return False

def check_app_response():
    """Check if the main app loads without errors"""
    try:
        response = requests.get('http://localhost:8502', timeout=10)
        return response.status_code == 200 and 'Smart Document Assistant' in response.text
    except Exception as e:
        print(f"App response error: {e}")
        return False

def check_environment_variables():
    """Check critical environment variables"""
    critical_vars = [
        'OPENAI_API_KEY',
        'HF_HOME', 
        'TRANSFORMERS_CACHE',
        'TORCH_HOME'
    ]
    
    missing = []
    for var in critical_vars:
        if not os.getenv(var):
            missing.append(var)
    
    return missing

def check_model_cache():
    """Check if model cache directories exist and have content"""
    cache_dirs = [
        '/app/models/transformers',
        '/app/models/huggingface', 
        '/app/models/torch',
        '/app/cache/embeddings'
    ]
    
    status = {}
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            files = list(Path(cache_dir).rglob('*'))
            status[cache_dir] = f"Exists ({len(files)} files)"
        else:
            status[cache_dir] = "Missing"
    
    return status

def check_gpu_availability():
    """Check GPU availability"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        return gpu_available, gpu_count
    except ImportError:
        return False, 0

def restart_streamlit_components():
    """Try to restart Streamlit session state"""
    try:
        # Make a request that triggers initialization
        requests.post('http://localhost:8502', 
                     data={'clear_cache': 'true'}, 
                     timeout=30)
        return True
    except:
        return False

def main():
    print("🔍 AI-RAG System Diagnosis")
    print("=" * 50)
    
    # Check 1: Streamlit Health
    print("\n1. Streamlit Health Check:")
    if check_streamlit_health():
        print("   ✅ Streamlit is running and responsive")
    else:
        print("   ❌ Streamlit is not responding")
        return
    
    # Check 2: App Response
    print("\n2. Application Response Check:")
    if check_app_response():
        print("   ✅ Main app loads successfully")
    else:
        print("   ❌ Main app has loading issues")
    
    # Check 3: Environment Variables
    print("\n3. Environment Variables Check:")
    missing_vars = check_environment_variables()
    if missing_vars:
        print(f"   ⚠️ Missing variables: {', '.join(missing_vars)}")
    else:
        print("   ✅ All critical environment variables present")
    
    # Check 4: Model Cache
    print("\n4. Model Cache Check:")
    cache_status = check_model_cache()
    for path, status in cache_status.items():
        print(f"   {path}: {status}")
    
    # Check 5: GPU
    print("\n5. GPU Availability Check:")
    gpu_available, gpu_count = check_gpu_availability()
    if gpu_available:
        print(f"   ✅ GPU available ({gpu_count} devices)")
    else:
        print("   ⚠️ No GPU available (CPU mode)")
    
    print("\n" + "=" * 50)
    print("🔧 RECOMMENDED ACTIONS:")
    
    if missing_vars:
        print("• Set missing environment variables")
    
    if not any("transformers" in status for status in cache_status.values()):
        print("• Model cache appears empty - initialization will be slow")
        print("• Consider restarting container to re-download models")
    
    print("• Try accessing the app and check browser console for errors")
    print("• Check if the 'Process Documents' button triggers specific errors")

if __name__ == "__main__":
    main()