#!/usr/bin/env python3
"""
Quick Launch Script for Graph-R1 Agentic RAG Demo

Provides a simple way to launch the Graph-R1 demonstration with
environment validation and helpful startup information.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check if environment is properly configured."""
    print("🧠 Graph-R1 Agentic RAG Demo Launcher")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Error: Python 3.8+ required")
        return False
    
    # Check required packages
    package_checks = {
        'streamlit': 'streamlit',
        'openai': 'openai', 
        'torch': 'torch',
        'transformers': 'transformers',
        'sentence-transformers': 'sentence_transformers',  # Note: import name differs
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for display_name, import_name in package_checks.items():
        try:
            __import__(import_name)
            print(f"✅ {display_name}")
        except ImportError:
            missing_packages.append(display_name)
            print(f"❌ {display_name} (missing)")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Check environment variables
    print("\n🔑 Environment Variables:")
    openai_key = os.getenv('OPENAI_API_KEY')
    sf_username = os.getenv('SALESFORCE_USERNAME')
    sf_password = os.getenv('SALESFORCE_PASSWORD')
    sf_token = os.getenv('SALESFORCE_SECURITY_TOKEN')
    
    if openai_key:
        print("✅ OPENAI_API_KEY loaded")
    else:
        print("❌ OPENAI_API_KEY missing (required)")
        return False
    
    if sf_username and sf_password and sf_token:
        print("✅ Salesforce credentials loaded")
    else:
        print("⚠️ Salesforce credentials incomplete (optional)")
    
    return True

def launch_demo():
    """Launch the Graph-R1 demo application."""
    demo_file = Path(__file__).parent / "test_graph_r1_demo.py"
    
    if not demo_file.exists():
        print(f"❌ Demo file not found: {demo_file}")
        return False
    
    print(f"\n🚀 Launching Graph-R1 Demo...")
    print(f"📄 Demo file: {demo_file}")
    print(f"🌐 URL will open automatically in your browser")
    print(f"🛑 Press Ctrl+C to stop the demo")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(demo_file)
        ])
        return True
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to launch demo: {e}")
        return False

def main():
    """Main launcher function."""
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Launch demo
    print("\n✅ Environment check passed!")
    
    # Give user option to continue
    try:
        input("\nPress Enter to launch Graph-R1 demo (Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n🛑 Launch cancelled by user")
        sys.exit(0)
    
    success = launch_demo()
    
    if success:
        print("\n🎉 Graph-R1 demo session completed!")
    else:
        print("\n❌ Demo launch failed")
        sys.exit(1)

if __name__ == "__main__":
    main()