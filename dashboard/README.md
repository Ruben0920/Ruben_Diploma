# Interactive Multi-Modal Analysis Dashboard

A comprehensive web-based dashboard for analyzing political debate content using sentiment analysis, emotion detection, and topic classification.

## Features

### 🎭 Interactive Transcript Explorer
- **Full Debate Transcript**: View the complete transcript with speaker identification
- **Color-Coded Speakers**: Each speaker has a distinct color for easy identification
- **Real-time Analysis**: Hover over text segments to see sentiment, emotion, and topic analysis
- **Timing Information**: Each segment shows start and end times

### 👤 Speaker Profile Deep Dive
- **Speaker Selection**: Choose any speaker from the dropdown menu
- **Sentiment Timeline**: Interactive line chart showing sentiment changes over time
- **Emotion Distribution**: Pie chart displaying detected emotions and their frequencies
- **Topic Focus**: Horizontal bar chart showing which topics each speaker focused on
- **Detailed Statistics**: Comprehensive metrics including average sentiment, segment count, and speaking time

### ⚔️ Topic Polarization Face-Off
- **Topic Selection**: Analyze any of the 15 predefined topics
- **Sentiment Comparison**: Bar chart comparing how different speakers feel about the same topic
- **Most Polarizing Statements**: View the top 3 most positive and negative statements for each topic
- **Speaker Analysis**: Understand how each speaker approaches specific topics

### 📊 Overall Debate Analysis
- **Interactive Heatmap**: Color-coded sentiment matrix for all topics and speakers
- **Key Performance Indicators**: Speaking time, average sentiment, and emotion statistics
- **Summary Metrics**: Total segments, speakers, topics, and duration
- **Global Emotion Distribution**: Overall emotion patterns across all speakers

## Installation

1. **Navigate to the dashboard directory:**
   ```bash
   cd dashboard
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the output directory is accessible:**
   The dashboard expects to find analysis data in the `../output/` directory relative to the dashboard folder.

## Usage

### Starting the Dashboard

1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your web browser** and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

### Navigation

- Use the **sidebar navigation** to switch between the four main sections
- Each section provides different insights into the debate analysis
- Interactive charts allow you to hover for detailed information

### Data Sources

The dashboard automatically loads and parses the following files from the output directory:

- `gender_topic_sentiment_report.json` - Speaker-level sentiment and emotion analysis
- `enhanced_polarization_report.json` - Segment-by-segment detailed analysis
- `speakers_with_embeddings.json` - Speaker diarization and timing information
- `transcription_with_timestamps.json` - Word-level transcription with timing

## Technical Details

### Architecture
- **Backend**: Python with Streamlit framework
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas for data manipulation and analysis
- **Caching**: Streamlit caching for improved performance

### Data Flow
1. **DataLoader** class parses JSON files and creates structured DataFrames
2. **Chart Functions** generate interactive Plotly visualizations
3. **Section Renderers** organize content into logical sections
4. **Streamlit Interface** provides the web-based user experience

### Performance Features
- **Data Caching**: Analysis data is cached to avoid repeated file parsing
- **Lazy Loading**: Charts are generated only when needed
- **Responsive Design**: Dashboard adapts to different screen sizes

## Customization

### Adding New Visualizations
1. Create a new chart function in `app.py`
2. Add the visualization to the appropriate section renderer
3. Update the navigation if needed

### Modifying Data Sources
1. Update the `DataLoader` class in `data_loader.py`
2. Add new parsing methods for additional file formats
3. Modify the data structure as needed

### Styling Changes
- Edit the CSS in the `st.markdown()` section of `app.py`
- Modify color schemes in the `get_speaker_color()` and `get_sentiment_color()` functions

## Troubleshooting

### Common Issues

1. **Data Loading Errors**
   - Ensure the output directory contains all required JSON files
   - Check file permissions and paths
   - Verify JSON file format and structure

2. **Chart Display Issues**
   - Refresh the browser page
   - Check browser console for JavaScript errors
   - Ensure all dependencies are properly installed

3. **Performance Issues**
   - Large datasets may take time to load initially
   - Use the caching features to improve subsequent loads
   - Consider reducing data size for very large files

### Error Messages

- **"Failed to load data"**: Check output directory path and file availability
- **"No transcript data available"**: Verify enhanced_polarization_report.json exists
- **"No speaker data available"**: Check gender_topic_sentiment_report.json

## Future Enhancements

- **Real-time Updates**: Live data streaming from the analysis pipeline
- **Export Functionality**: Download charts and data as PDF/CSV
- **Advanced Filtering**: Filter by time ranges, sentiment thresholds, or topics
- **Comparative Analysis**: Side-by-side comparison of multiple debates
- **User Authentication**: Multi-user support with role-based access

## Support

For technical support or feature requests, please refer to the main project documentation or create an issue in the project repository.

## License

This dashboard is part of the larger multi-modal analysis pipeline project. Please refer to the main project license for usage terms.
