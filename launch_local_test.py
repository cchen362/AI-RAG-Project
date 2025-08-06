#!/usr/bin/env python3
"""
Quick launcher for local testing without Salesforce authentication
"""
import os
import subprocess
import sys
import time

def launch_app():
    """Launch the app in testing mode"""
    print("🚀 Launching AI-RAG-Project for local testing...")
    print("=" * 50)
    
    # Set environment variables to skip problematic components
    env = os.environ.copy()
    env['SKIP_SALESFORCE'] = 'true'
    env['CPU_ONLY_MODE'] = 'true'
    
    # Start streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_rag_app.py",
        "--server.port", "8501",
        "--server.headless", "false"
    ]
    
    print(f"📝 Running: {' '.join(cmd)}")
    print("🌐 App will be available at: http://localhost:8501")
    print("⏱️ Please wait for initialization (may take 1-2 minutes)...")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    launch_app()