"""
Launch GUI Test Interface

Simple launcher script for the agentic RAG GUI testing interface.
"""

import subprocess
import sys
import os

def launch_gui_test():
    """Launch the GUI testing interface"""
    print("*** Launching Agentic RAG GUI Testing Interface ***")
    print("=" * 50)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("[OK] Streamlit available")
    except ImportError:
        print("[ERROR] Streamlit not found. Please install with: pip install streamlit")
        return
    
    # Check if gui_test_interface.py exists
    gui_file = os.path.join(os.path.dirname(__file__), 'gui_test_interface.py')
    if not os.path.exists(gui_file):
        print(f"[ERROR] GUI test interface not found: {gui_file}")
        return
    
    print("[OK] GUI test interface found")
    print("[LAUNCH] Starting Streamlit server...")
    print()
    print("[BROWSER] The interface will open in your default browser")
    print("[URL] If it doesn't open automatically, go to: http://localhost:8501")
    print()
    print("[FEATURES] Available capabilities:")
    print("   - Interactive query testing with real-time reasoning visualization")
    print("   - Predefined test scenarios")
    print("   - Performance analytics dashboard")
    print("   - Agent memory inspection")
    print()
    print("[STOP] Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            gui_file,
            '--server.port=8501',
            '--server.address=localhost'
        ])
    except KeyboardInterrupt:
        print("\n[STOP] GUI test interface stopped")
    except Exception as e:
        print(f"[ERROR] Failed to launch interface: {e}")

if __name__ == "__main__":
    launch_gui_test()