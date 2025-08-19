#!/bin/bash

# Interactive Multi-Modal Analysis Dashboard Launcher
# This script sets up and runs the Streamlit dashboard

echo "🎭 Interactive Multi-Modal Analysis Dashboard"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the dashboard directory."
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit, plotly, pandas" 2>/dev/null; then
    echo "⚠️  Some dependencies are missing. Installing requirements..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install requirements. Please check your Python environment."
        exit 1
    fi
    echo "✅ Dependencies installed successfully!"
else
    echo "✅ All dependencies are already installed!"
fi

# Check if output directory exists
if [ ! -d "../output" ]; then
    echo "⚠️  Warning: ../output directory not found. The dashboard may not work properly."
    echo "   Make sure you have run the analysis pipeline first."
fi

# Test data loading
echo "🧪 Testing data loading..."
python3 test_data_loading.py
if [ $? -ne 0 ]; then
    echo "⚠️  Data loading test failed. The dashboard may not work properly."
    echo "   Continuing anyway..."
fi

echo ""
echo "🚀 Starting the dashboard..."
echo "   The dashboard will open in your default web browser."
echo "   If it doesn't open automatically, navigate to: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the dashboard."
echo ""

# Run the dashboard using python3 -m streamlit
python3 -m streamlit run app.py
