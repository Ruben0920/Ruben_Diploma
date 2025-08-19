# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY face_recognition.py .

# Command to run the script
CMD ["python", "face_recognition.py"]

