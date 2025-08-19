#!/usr/bin/env python3
"""
Python-based launcher for the Interactive Multi-Modal Analysis Dashboard
This script handles PATH issues and provides a more reliable launch method
"""

import sys
import os
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    try:
        import streamlit
        import plotly
        import pandas
        print("✅ All dependencies are already installed!")
        return True
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("Installing requirements...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False

def test_data_loading():
    """Test the data loading functionality"""
    print("🧪 Testing data loading...")
    
    try:
        from test_data_loading import test_data_loading
        success = test_data_loading()
        if success:
            print("✅ Data loading test passed!")
            return True
        else:
            print("⚠️  Data loading test failed, but continuing...")
            return True
    except Exception as e:
        print(f"⚠️  Data loading test error: {e}")
        print("Continuing anyway...")
        return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n🚀 Starting the dashboard...")
    print("   The dashboard will open in your default web browser.")
    print("   If it doesn't open automatically, navigate to: http://localhost:8501")
    print("\n   Press Ctrl+C to stop the dashboard.")
    print("=" * 60)
    
    # Launch the dashboard
    try:
        # Use python -m streamlit to avoid PATH issues
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Try to open the browser
        try:
            webbrowser.open("http://localhost:8501")
            print("🌐 Browser opened automatically!")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("   Please manually navigate to: http://localhost:8501")
        
        print(f"\n📊 Dashboard is running with PID: {process.pid}")
        print("   To stop the dashboard, press Ctrl+C or kill the process")
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        if 'process' in locals():
            process.terminate()
            process.wait()
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🎭 Interactive Multi-Modal Analysis Dashboard")
    print("==============================================")
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found. Please run this script from the dashboard directory.")
        return False
    
    # Check if output directory exists
    if not Path("../output").exists():
        print("⚠️  Warning: ../output directory not found.")
        print("   Make sure you have run the analysis pipeline first.")
        print("   The dashboard may not work properly.")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Test data loading
    if not test_data_loading():
        return False
    
    # Launch dashboard
    return launch_dashboard()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Launcher stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
