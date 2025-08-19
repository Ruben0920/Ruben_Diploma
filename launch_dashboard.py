#!/usr/bin/env python3
"""
Python-based launcher for the Streamlit dashboard
Provides robust startup and graceful shutdown
"""

import subprocess
import sys
import time
import signal
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import plotly
        import pandas
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages: pip install -r dashboard/requirements.txt")
        return False

def test_data_loading():
    """Test if the dashboard can load data successfully"""
    try:
        sys.path.append(str(Path(__file__).parent / "dashboard"))
        from data_loader import DataLoader
        
        data_loader = DataLoader()
        speakers = data_loader.get_speakers()
        topics = data_loader.get_topics()
        
        if speakers and topics:
            print(f"Data loaded successfully: {len(speakers)} speakers, {len(topics)} topics")
            return True
        else:
            print("Warning: Limited data available")
            return True
    except Exception as e:
        print(f"Data loading test failed: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard with error handling"""
    dashboard_dir = Path(__file__).parent / "dashboard"
    
    if not dashboard_dir.exists():
        print("Error: Dashboard directory not found")
        return False
    
    if not (dashboard_dir / "app.py").exists():
        print("Error: Dashboard app.py not found")
        return False
    
    print("Starting Streamlit dashboard...")
    print(f"Dashboard directory: {dashboard_dir}")
    
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"],
            cwd=dashboard_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Dashboard process started")
        print("Waiting for dashboard to be ready...")
        
        time.sleep(3)
        
        if process.poll() is None:
            print("Dashboard is running successfully!")
            print("Opening browser...")
            
            try:
                webbrowser.open("http://localhost:8501")
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print("Please open: http://localhost:8501")
            
            print("Dashboard launched successfully!")
            print("Press Ctrl+C to stop the dashboard")
            
            try:
                process.wait()
            except KeyboardInterrupt:
                print("Stopping dashboard...")
                process.terminate()
                process.wait()
                print("Dashboard stopped")
            
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"Dashboard failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        return False

def main():
    """Main entry point"""
    print("Interactive Multi-Modal Analysis Dashboard Launcher")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    if not test_data_loading():
        print("Warning: Data loading test failed, but continuing...")
    
    success = launch_dashboard()
    
    if not success:
        print("Failed to launch dashboard")
        sys.exit(1)

if __name__ == "__main__":
    main()
