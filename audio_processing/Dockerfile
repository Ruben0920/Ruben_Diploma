# Use Python base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install PyTorch dependencies from the PyTorch-specific index URL
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from PyPI
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Set the default command
CMD ["python", "audio_processing.py"]
