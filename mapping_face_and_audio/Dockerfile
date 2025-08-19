FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 git && apt-get clean
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Copy the script
COPY combine_faces_and_speakers.py .

# Set default command
CMD ["python", "combine_faces_and_speakers.py"]
