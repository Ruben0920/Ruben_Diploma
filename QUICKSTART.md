# Quick Start Guide

## Interactive Multi-Modal Analysis Dashboard

Get up and running with the dashboard in minutes!

## Prerequisites

- Python 3.9 or higher
- 8GB+ RAM available
- Internet connection for package installation

## Installation

### 1. Install Python Dependencies
```bash
pip install -r dashboard/requirements.txt
```

### 2. Launch Dashboard

#### Option A: Python Launcher (Recommended)
```bash
python3 launch_dashboard.py
```

#### Option B: Shell Script
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

#### Option C: Manual Launch
```bash
cd dashboard
streamlit run app.py
```

## What You'll See

1. **Interactive Transcript Explorer**: Full debate transcript with sentiment analysis
2. **Speaker Profile Deep Dive**: Individual speaker analysis and statistics
3. **Topic Polarization Face-Off**: Topic-based sentiment comparison
4. **Overall Debate Analysis**: Comprehensive statistics and heatmaps

## Troubleshooting

### Dashboard Not Loading
- Check Python dependencies: `pip list | grep streamlit`
- Verify data files exist in `output/` directory
- Try manual launch: `cd dashboard && streamlit run app.py`

### Missing Dependencies
```bash
pip install streamlit plotly pandas numpy
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

## Data Requirements

The dashboard automatically loads from:
- `output/enhanced_polarization_report.json`
- `output/gender_topic_sentiment_report.json`
- `output/speakers_with_embeddings.json`

## Next Steps

1. Explore the transcript with interactive tooltips
2. Analyze individual speaker sentiment patterns
3. Compare topic polarization across speakers
4. Review overall debate statistics

## Support

- Check the main README.md for detailed documentation
- Run `python3 dashboard/test_data_loading.py` to verify data
- Use `python3 dashboard/demo_features.py` to see capabilities

Happy analyzing!
