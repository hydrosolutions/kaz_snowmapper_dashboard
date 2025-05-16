# Use Miniconda as the base image
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
COPY ./requirements.yml /app/requirements.yml
COPY ./pip_requirements.txt /app/pip_requirements.txt

# Install Conda dependencies
RUN conda env create -f /app/requirements.yml && \
    conda clean --all -y

# Activate the Conda environment and install pip dependencies
RUN /bin/bash -c "source activate mcass_kaz && \
    pip install --no-cache-dir -r /app/pip_requirements.txt"

# Export the Conda environment to ensure reproducibility
RUN conda env export > /app/exported_environment.yml

# Final stage: Use a smaller base image for runtime
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy the Conda environment from the builder stage
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app /app

# Set environment variables
ENV PATH="/conda/envs/mcass_kaz/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PROJ_LIB="/conda/envs/mcass_kaz/share/proj" \
    PROJ_DATA="/conda/envs/mcass_kaz/share/proj" \
    GDAL_DATA="/conda/envs/mcass_kaz/share/gdal"

# Activate the Conda environment by default
SHELL ["/bin/bash", "-c"]

RUN source activate mcass_kaz && echo "===== PATHS =====" && \
    ls -la | grep conda && \
    echo "===== CONDA ENV =====" && \
    conda info --envs
    
# Ensure PROJ data is available in conda environment
RUN bash -c source activate mcass_kaz && \
    mkdir -p /conda/envs/mcass_kaz/share/proj && \
    mkdir -p /conda/envs/mcass_kaz/share/gdal && \
    cp -r /usr/share/proj/* /conda/envs/mcass_kaz/share/proj/ 2>/dev/null || true && \
    cp -r /usr/share/gdal/* /conda/envs/mcass_kaz/share/gdal/ 2>/dev/null || true

# Copy the rest of the application
COPY . /app/

# Expose the port for the dashboard
EXPOSE 5007

# Set the default command to run the Panel dashboard
CMD ["bash", "-c", "source activate mcass_kaz && panel serve /app/dashboard/snowmapper.py --port 5007 --allow-websocket-origin=kaz.snowmapper.ch --allow-websocket-origin=0.0.0.0 --address 0.0.0.0"]