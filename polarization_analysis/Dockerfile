# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY polarization_requirements.txt .
RUN pip install --no-cache-dir -r polarization_requirements.txt

# Copy the polarization analysis scripts
COPY enhanced_polarization_analysis.py .
COPY generate_narrative_report.py .
COPY test_narrative.py .

# Create output directory
RUN mkdir -p /app/output

# Command to run the enhanced script
CMD ["python", "enhanced_polarization_analysis.py"] 