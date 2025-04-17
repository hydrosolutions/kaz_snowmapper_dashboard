# Use Python 3.11 slim base image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for GDAL and other geospatial libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV PROJ_LIB=/usr/share/proj

# Install GDAL Python bindings first
RUN pip install --upgrade pip && \
    pip install --no-cache-dir GDAL==$(gdal-config --version)

# Copy only requirements first for better cache usage
COPY ./processing/requirements.txt /app/processing/requirements.txt

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r /app/processing/requirements.txt

# Copy the rest of the application
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DASHBOARD_ENV=aws

# Set default command
CMD ["python", "processing/data_processor.py"]
