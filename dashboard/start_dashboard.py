#!/usr/bin/env python3
"""
Simple Dashboard Launcher - This will definitely work!
"""

import subprocess
import sys
import time
import webbrowser
import os

def main():
    print("🎭 Interactive Multi-Modal Analysis Dashboard")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: Please run this from the dashboard directory")
        return
    
    # Check dependencies
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} is ready")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} installed")
    
    # Test data loading
    print("🧪 Testing data...")
    try:
        from test_data_loading import test_data_loading
        test_data_loading()
        print("✅ Data loading test passed!")
    except Exception as e:
        print(f"⚠️  Data test warning: {e}")
    
    print("\n🚀 Starting dashboard...")
    print("   This will open in your browser automatically")
    print("   If not, go to: http://localhost:8501")
    print("\n   Press Ctrl+C to stop")
    print("=" * 50)
    
    # Launch dashboard
    try:
        # Use the full path to avoid PATH issues
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Start the process
        process = subprocess.Popen(cmd)
        
        # Wait for server to start
        time.sleep(5)
        
        # Try to open browser
        try:
            webbrowser.open("http://localhost:8501")
            print("🌐 Browser opened!")
        except:
            print("⚠️  Please open: http://localhost:8501")
        
        print(f"📊 Dashboard running (PID: {process.pid})")
        print("   Press Ctrl+C to stop")
        
        # Wait for completion
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("✅ Dashboard stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
