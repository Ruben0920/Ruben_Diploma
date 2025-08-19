# Ruben_Diploma

A comprehensive pipeline for analyzing political debates through face recognition, speaker diarization, and polarization analysis.

## Overview

This project implements an end-to-end pipeline for analyzing political debate videos to extract insights about:
- **Face Recognition**: Identifies and tracks individuals throughout debates
- **Speaker Diarization**: Separates and identifies different speakers
- **Audio Processing**: Processes audio for speech analysis
- **Polarization Analysis**: Analyzes sentiment and polarization in political discourse
- **Dashboard**: Interactive web interface for exploring results

## System Architecture

The Ruben_Diploma project implements a sophisticated, containerized microservices architecture designed for high-performance parallel video processing. The system leverages Docker containers, parallel execution, and modular design to efficiently analyze political debate videos through multiple AI-powered analysis stages.

### High-Level Architecture Overview

The pipeline follows a **multi-stage, parallel processing architecture** that processes videos through four core analysis services:

1. **Audio Processing Service** - Speaker diarization and transcription
2. **Face Recognition Service** - Facial detection and identity tracking  
3. **Mapping Service** - Correlation of faces with speakers
4. **Polarization Analysis Service** - Sentiment and political discourse analysis

### Container-Based Microservices Design

The system uses **Docker containers** for each analysis service, providing:

- **Isolation**: Each service runs in its own container with dedicated resources
- **Scalability**: Services can be scaled independently based on workload
- **Portability**: Consistent execution environment across different systems
- **Resource Management**: Controlled CPU and memory allocation per service

#### Docker Service Configuration

```yaml
# Resource allocation per service
audio_processing:     2GB RAM, 1.0 CPU
face_recognition:     3GB RAM, 1.5 CPU  
mapping_face_and_audio: 2GB RAM, 1.0 CPU
polarization_analysis: 2GB RAM, 1.0 CPU
```

Each service container:
- Mounts shared input/output volumes for data exchange
- Runs with health checks and automatic restart policies
- Implements graceful shutdown handling
- Uses environment variables for configuration

### Parallel Processing Orchestration

The system implements **two-level parallelization** for maximum efficiency:

#### Level 1: Parallel Video Processing
- Multiple videos processed simultaneously using `ThreadPoolExecutor`
- Configurable concurrency limit (default: 4 concurrent videos)
- Independent processing pipelines per video

#### Level 2: Parallel Service Execution
- Within each video, services run concurrently where possible
- Dependency-aware execution (some services require outputs from others)
- Configurable service concurrency (default: 2 concurrent services per video)

### Pipeline Orchestration Components

#### 1. **run_pipeline.py** - Master Control Script
- **Entry Point**: Main pipeline execution controller
- **Video Discovery**: Scans input directory for video files
- **Metadata Extraction**: Generates video metadata using OpenCV
- **Sequential Service Execution**: Manages service dependencies and execution order
- **Output Validation**: Verifies service outputs and handles failures gracefully

#### 2. **parallel_orchestrator.py** - Advanced Orchestration Engine
- **Docker API Integration**: Direct container management via Docker SDK
- **Resource Monitoring**: System resource validation before processing
- **Dynamic Scaling**: Adjusts concurrency based on available resources
- **Fault Tolerance**: Comprehensive error handling and recovery
- **Progress Tracking**: Real-time progress monitoring with TQDM

### Data Flow Architecture

```
Input Videos → Metadata Extraction → Parallel Service Execution → Output Aggregation
     ↓                    ↓                        ↓                    ↓
  Video Files        JSON Metadata         Containerized Services    Analysis Results
     ↓                    ↓                        ↓                    ↓
  Volume Mount      Processing Logs        Parallel Execution      Structured Output
```

### Service Dependencies and Execution Order

The pipeline implements **intelligent dependency management**:

1. **Audio Processing** (Independent)
   - Extracts audio from video
   - Performs speaker diarization
   - Generates transcription with timestamps

2. **Face Recognition** (Depends on: Audio Processing)
   - Requires speaker diarization output
   - Detects and tracks faces throughout video
   - Generates face embeddings and metadata

3. **Mapping Service** (Depends on: Audio + Face Recognition)
   - Correlates detected faces with identified speakers
   - Creates unified speaker-face mappings

4. **Polarization Analysis** (Depends on: Audio Processing)
   - Analyzes transcribed text for sentiment
   - Generates political polarization metrics
   - Creates visualization outputs

### Resource Management and Optimization

#### Memory Management
- **Container Memory Limits**: Prevents memory overflow
- **Shared Volume Mounts**: Efficient data exchange between services
- **Garbage Collection**: Automatic cleanup of temporary resources

#### CPU Optimization
- **CPU Quota System**: Fair resource allocation across containers
- **Concurrency Control**: Prevents system overload
- **Load Balancing**: Distributes processing across available cores

#### Storage Optimization
- **Read-Only Input**: Prevents accidental modification of source videos
- **Structured Output**: Organized directory hierarchy for results
- **Metadata Tracking**: Comprehensive logging and audit trails

### Fault Tolerance and Error Handling

The system implements **robust error handling**:

- **Service-Level Resilience**: Individual service failures don't crash the pipeline
- **Graceful Degradation**: Partial results saved even when some services fail
- **Comprehensive Logging**: Detailed error reporting and debugging information
- **Automatic Recovery**: Container restart policies and health checks
- **Dependency Validation**: Pre-execution checks for required inputs

### Performance Characteristics

#### Scalability
- **Horizontal Scaling**: Add more containers for increased throughput
- **Vertical Scaling**: Adjust resource limits per service
- **Load Distribution**: Parallel processing across multiple videos

#### Efficiency Metrics
- **Processing Speed**: ~2-4x faster than sequential processing
- **Resource Utilization**: Optimal CPU and memory usage
- **Throughput**: Configurable based on system capabilities

### Monitoring and Observability

- **Real-time Progress**: Live progress bars and status updates
- **Resource Monitoring**: CPU, memory, and disk usage tracking
- **Service Health**: Container health checks and status reporting
- **Performance Metrics**: Processing time and success rate statistics

This architecture enables the system to efficiently process large volumes of political debate videos while maintaining reliability, scalability, and performance across diverse computing environments.

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