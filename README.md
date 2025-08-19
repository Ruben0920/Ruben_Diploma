# Ruben_Diploma

A comprehensive pipeline for analyzing political debates through face recognition, speaker diarization, and polarization analysis.

## Overview

This project implements an end-to-end pipeline for analyzing political debate videos to extract insights about:
- **Face Recognition**: Identifies and tracks individuals throughout debates
- **Speaker Diarization**: Separates and identifies different speakers
- **Audio Processing**: Processes audio for speech analysis
- **Polarization Analysis**: Analyzes sentiment and polarization in political discourse
- **Dashboard**: Interactive web interface for exploring results

## Project Structure

```
Ruben_Diploma/
├── audio_processing/          # Audio processing pipeline
├── dashboard/                 # Web dashboard application
├── face_recognition/          # Face detection and recognition
├── mapping_face_and_audio/    # Combines face and speaker data
├── pipeline_orchestrator/     # Main pipeline coordination
├── polarization_analysis/     # Sentiment and polarization analysis
├── utilities/                 # Helper utilities
├── input/                     # Input video files
├── output/                    # Analysis results
└── docker-compose.yml         # Docker orchestration
```

## Features

- **Multi-modal Analysis**: Combines video, audio, and text analysis
- **Scalable Pipeline**: Docker-based architecture for easy deployment
- **Interactive Dashboard**: Web-based visualization of results
- **Parallel Processing**: Efficient handling of multiple video files
- **Comprehensive Output**: Detailed reports and visualizations

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- FFmpeg (for video processing)
- Hugging Face account and token

### Setup

1. **Get your Hugging Face token**
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with read access
   - Copy the token

2. **Set environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your Hugging Face token
   export HUGGING_FACE_TOKEN=your_token_here
   ```

### Running the Pipeline

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Ruben_Diploma.git
   cd Ruben_Diploma
   ```

2. **Place your video files**
   - Add debate videos to `input/videos/` directory

3. **Run the pipeline**
   ```bash
   python run_pipeline.py
   ```

4. **Launch the dashboard**
   ```bash
   python launch_dashboard.py
   ```

### Docker Deployment

```bash
docker-compose up -d
```

## Configuration

The pipeline can be configured through environment variables and configuration files in each module directory.

## Output

Results are stored in the `output/` directory with the following structure:
- `face_recognition/`: Detected faces and identities
- `speaker_diarization/`: Speaker separation and identification
- `polarization_analysis/`: Sentiment and polarization metrics
- `enhanced_visualizations/`: Charts and interactive visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is part of academic research. Please contact the author for usage permissions.

## Contact

For questions or collaboration, please contact the project maintainer. 