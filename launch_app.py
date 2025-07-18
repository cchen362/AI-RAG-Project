import subprocess
import sys
import os

def main():
    """Launch the Streamlit RAG application."""
    
    print("🚀 Starting Smart Document Assistant...")
    print("="*50)
    
    # Check if required packages are installed
    required_packages = ['streamlit', 'plotly', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Set environment variables
    os.environ.setdefault('STREAMLIT_SERVER_PORT', '8501')
    os.environ.setdefault('STREAMLIT_SERVER_ADDRESS', 'localhost')
    
    # Launch Streamlit
    print("\n🌐 Starting web interface...")
    print("📱 Open your browser and go to: http://localhost:8501")
    print("💡 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_rag_app.py",
            "--server.headless", "true",
            "--server.fileWatcherType", "none"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main()