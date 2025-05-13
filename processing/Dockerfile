# Use Python 3.11 slim base image for smaller size
#FROM python:3.11-slim
# Use a slim ppython image with miniconda installed
FROM continuumio/miniconda3:latest AS builder

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

# Copy the Conda environment file and pip requirements file
COPY ../requirements.yml /app/requirements.yml
COPY ../pip_requirements.txt /app/pip_requirements.txt

# Install Conda dependencies
RUN conda env create -f /app/requirements.yml && \
    conda clean --all -y

# Activate the Conda environment and install pip dependencies
RUN /bin/bash -c "source activate mcass_kaz && \
    pip install --no-cache-dir -r /app/pip_requirements.txt"

# Export the Conda environment to ensure reproducibility
RUN conda env export > /app/exported_environment.yml

# Install GDAL Python bindings first
#RUN pip install --upgrade pip && \
#    pip install --no-cache-dir GDAL==$(gdal-config --version)

# Create necessary directories
RUN mkdir -p /app/logs

# Final stage: Use a smaller base image for runtime
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy the Conda environment from the builder stage
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app /app

# Set environment variables
ENV PATH="/opt/conda/envs/mcass_kaz/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DASHBOARD_ENV=aws 

# Activate the Conda environment by default
SHELL ["/bin/bash", "-c", "source activate mcass_kaz &&"]

# Copy the rest of the application
COPY . /app/

# Set default command
CMD ["python", "processing/data_processor.py"]
