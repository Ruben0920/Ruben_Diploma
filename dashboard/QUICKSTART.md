# 🚀 Quick Start Guide

Get your Interactive Multi-Modal Analysis Dashboard running in under 5 minutes!

## Prerequisites

- Python 3.7+ installed
- Access to the output directory with analysis data

## Option 1: Python Launcher (Recommended - Fixes PATH Issues)

```bash
# Navigate to dashboard directory
cd dashboard

# Run the Python launcher (handles PATH issues automatically)
python3 launch_dashboard.py
```

## Option 2: Shell Script Launcher

```bash
# Navigate to dashboard directory
cd dashboard

# Run the launcher script
./run_dashboard.sh
```

## Option 3: Manual Setup

```bash
# 1. Navigate to dashboard directory
cd dashboard

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Test data loading
python3 test_data_loading.py

# 4. Launch dashboard (use python -m to avoid PATH issues)
python3 -m streamlit run app.py
```

## What Happens Next

1. **Browser Opens**: The dashboard will automatically open in your default browser
2. **URL**: If it doesn't open automatically, go to `http://localhost:8501`
3. **Navigation**: Use the sidebar to explore the four main sections

## Dashboard Sections

### 📝 Interactive Transcript Explorer
- View the complete debate transcript
- Color-coded by speaker
- Hover over text for detailed analysis

### 👤 Speaker Profile Deep Dive
- Select any speaker from the dropdown
- View sentiment timeline, emotion distribution, and topic focus
- Detailed statistics for each speaker

### ⚔️ Topic Polarization Face-Off
- Choose any topic to analyze
- Compare speaker sentiments on the same topic
- View most polarizing statements

### 📊 Overall Debate Analysis
- Interactive sentiment heatmap
- Key performance indicators
- Summary statistics and global patterns

## Troubleshooting

### "streamlit: command not found" Error
This is a common PATH issue. Use one of these solutions:

1. **Use the Python launcher** (recommended):
   ```bash
   python3 launch_dashboard.py
   ```

2. **Use python -m streamlit**:
   ```bash
   python3 -m streamlit run app.py
   ```

3. **Add to PATH** (if you want to use `streamlit` directly):
   ```bash
   export PATH="$HOME/Library/Python/3.9/bin:$PATH"
   ```

### "Failed to load data"
- Ensure you're in the `dashboard` directory
- Check that `../output/` contains the required JSON files
- Verify file permissions

### "Module not found" errors
- Run `pip3 install -r requirements.txt`
- Ensure you're using Python 3.7+

### Charts not displaying
- Refresh the browser page
- Check browser console for errors
- Ensure all dependencies are installed

## Data Files Required

The dashboard expects these files in the `../output/` directory:
- `gender_topic_sentiment_report.json`
- `enhanced_polarization_report.json`
- `speakers_with_embeddings.json`
- `transcription_with_timestamps.json`

## Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Run `python3 test_data_loading.py` to diagnose data issues
- Use `python3 demo_features.py` to see all features
- Ensure your analysis pipeline has completed successfully

---

**🎉 You're all set! The Python launcher (`python3 launch_dashboard.py`) is the most reliable method!**
