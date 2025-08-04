#!/usr/bin/env python3
"""
Emergency Fix Script - Attempt to fix the container from user space
This tries to work around the Docker permission limitations
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

def try_create_directories():
    """Try to create necessary directories"""
    directories = [
        '/app/models/transformers',
        '/app/models/huggingface', 
        '/app/models/torch',
        '/app/cache/embeddings',
        '/tmp/models/transformers',
        '/tmp/models/huggingface',
        '/tmp/models/torch'
    ]
    
    created = []
    failed = []
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            if Path(directory).exists():
                created.append(directory)
            else:
                failed.append(directory)
        except Exception as e:
            failed.append(f"{directory} ({e})")
    
    return created, failed

def try_set_environment_variables():
    """Try to set environment variables for current session"""
    env_vars = {
        'TRANSFORMERS_CACHE': '/tmp/models/transformers',
        'HF_HOME': '/tmp/models/huggingface',
        'TORCH_HOME': '/tmp/models/torch',
        'HF_HUB_CACHE': '/tmp/models/huggingface',
        'HUGGINGFACE_HUB_CACHE': '/tmp/models/huggingface'
    }
    
    set_vars = []
    for key, value in env_vars.items():
        try:
            os.environ[key] = value
            # Also try to export for shell
            os.system(f'export {key}={value}')
            set_vars.append(f"{key}={value}")
        except:
            pass
    
    return set_vars

def create_env_file():
    """Create a .env file in accessible locations"""
    env_content = """# Emergency environment variables
TRANSFORMERS_CACHE=/tmp/models/transformers
HF_HOME=/tmp/models/huggingface
TORCH_HOME=/tmp/models/torch
HF_HUB_CACHE=/tmp/models/huggingface
HUGGINGFACE_HUB_CACHE=/tmp/models/huggingface
MODEL_DEVICE=cpu
DOCKER_PRELOADED_MODELS=false
"""
    
    locations = ['/tmp/.env', '~/.env', './.env']
    created = []
    
    for location in locations:
        try:
            with open(os.path.expanduser(location), 'w') as f:
                f.write(env_content)
            created.append(location)
        except:
            pass
    
    return created

def try_install_missing_packages():
    """Try to install any missing Python packages"""
    packages = [
        'python-dotenv',
        'requests',
        'transformers',
        'torch',
        'sentence-transformers'
    ]
    
    installed = []
    failed = []
    
    for package in packages:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', package], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                installed.append(package)
            else:
                failed.append(f"{package} ({result.stderr[:100]})")
        except Exception as e:
            failed.append(f"{package} ({e})")
    
    return installed, failed

def test_python_imports():
    """Test if critical Python packages can be imported"""
    imports = {
        'torch': 'import torch',
        'transformers': 'import transformers', 
        'sentence_transformers': 'import sentence_transformers',
        'openai': 'import openai',
        'streamlit': 'import streamlit'
    }
    
    working = []
    broken = []
    
    for name, import_stmt in imports.items():
        try:
            exec(import_stmt)
            working.append(name)
        except Exception as e:
            broken.append(f"{name} ({e})")
    
    return working, broken

def create_minimal_openai_config():
    """Create a minimal OpenAI configuration"""
    config_content = """# Minimal OpenAI config - placeholder
# This file serves as a placeholder until admin adds real API key
import os
import warnings

# Placeholder API key that will trigger a clear error message
OPENAI_API_KEY_PLACEHOLDER = "sk-placeholder-key-admin-needs-to-set-real-key"

def get_openai_key():
    key = os.getenv('OPENAI_API_KEY', OPENAI_API_KEY_PLACEHOLDER)
    if key == OPENAI_API_KEY_PLACEHOLDER:
        warnings.warn("OpenAI API key not set - administrator needs to configure")
    return key
"""
    
    try:
        with open('/tmp/openai_config.py', 'w') as f:
            f.write(config_content)
        return True
    except:
        return False

def main():
    print("🚀 Emergency Fix Attempt")
    print("=" * 40)
    
    # Step 1: Create directories
    print("\n1. Creating directories...")
    created_dirs, failed_dirs = try_create_directories()
    print(f"   ✅ Created: {len(created_dirs)} directories")
    for d in created_dirs[:3]:  # Show first 3
        print(f"      - {d}")
    if len(created_dirs) > 3:
        print(f"      ... and {len(created_dirs) - 3} more")
    
    if failed_dirs:
        print(f"   ❌ Failed: {len(failed_dirs)} directories")
    
    # Step 2: Set environment variables
    print("\n2. Setting environment variables...")
    set_vars = try_set_environment_variables()
    print(f"   ✅ Set {len(set_vars)} variables")
    for var in set_vars[:3]:
        print(f"      - {var}")
    
    # Step 3: Create .env files
    print("\n3. Creating .env files...")
    env_files = create_env_file()
    print(f"   ✅ Created {len(env_files)} .env files")
    for f in env_files:
        print(f"      - {f}")
    
    # Step 4: Test Python imports
    print("\n4. Testing Python imports...")
    working, broken = test_python_imports()
    print(f"   ✅ Working: {', '.join(working)}")
    if broken:
        print(f"   ❌ Broken: {', '.join(broken[:2])}")  # Show first 2
    
    # Step 5: Create OpenAI placeholder
    print("\n5. Creating OpenAI placeholder...")
    if create_minimal_openai_config():
        print("   ✅ OpenAI placeholder created")
    else:
        print("   ❌ Failed to create OpenAI placeholder")
    
    print("\n" + "=" * 40)
    print("🎯 EMERGENCY FIX SUMMARY:")
    print(f"• Created {len(created_dirs)} cache directories")
    print(f"• Set {len(set_vars)} environment variables") 
    print(f"• Created {len(env_files)} configuration files")
    print("• Created OpenAI placeholder configuration")
    
    print("\n🔧 NEXT STEPS:")
    print("• Try uploading a document to test initialization")
    print("• If it still fails, the administrator needs to add the OpenAI API key")
    print("• Models will download on first use (may take 2-3 minutes)")

if __name__ == "__main__":
    main()