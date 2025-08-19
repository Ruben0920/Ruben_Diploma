FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    curl \
    jq \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the orchestrator script
COPY parallel_orchestrator.py .

# Make the script executable
RUN chmod +x parallel_orchestrator.py

# Set default command
CMD ["python", "parallel_orchestrator.py"]
