# Use Miniconda as the base image for building the environment
FROM continuumio/miniconda3:latest AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required for GDAL and other geospatial libraries
# Ensure these are *build-time* dependencies if only needed for Conda package compilation,
# or provide data files needed later.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL during the BUILD phase if needed
# ENV GDAL_CONFIG=/usr/bin/gdal-config
# ENV PROJ_LIB=/usr/share/proj

# Copy the Conda environment file and pip requirements file
COPY ./requirements.yml /app/requirements.yml
COPY ./pip_requirements.txt /app/pip_requirements.txt

# Install Conda dependencies into a named environment
RUN conda env create -f /app/requirements.yml -n mcass_kaz && \
    conda clean --all -y

# Activate the Conda environment and install pip dependencies
# Use runuser or similar if permissions are an issue, but bash -c with source is standard
RUN /bin/bash -c "source activate mcass_kaz && \
    pip install --no-cache-dir -r /app/pip_requirements.txt"

# --- Final Stage ---
# Use a smaller base image for runtime, keeping a clean Conda installation
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy ONLY the specific Conda environment from the builder stage
# Copy it into the standard environments directory of the final image's Conda installation
COPY --from=builder /opt/conda/envs/mcass_kaz /opt/conda/envs/mcass_kaz
# Optionally copy exported_environment.yml if you need it in the final image
# COPY --from=builder /app/exported_environment.yml /app/exported_environment.yml


# Set environment variables for the runtime environment
ENV PATH="/opt/conda/envs/mcass_kaz/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PROJ_LIB="/opt/conda/envs/mcass_kaz/share/proj" \
    PROJ_DATA="/opt/conda/envs/mcass_kaz/share/proj" \
    GDAL_DATA="/opt/conda/envs/mcass_kaz/share/gdal"

# Set the shell for RUN and CMD instructions to allow 'source'
SHELL ["/bin/bash", "-c"]

# Ensure PROJ and GDAL data files are available in the Conda environment's share directory
# Copying from /usr/share from the builder stage where they were installed via apt
RUN bash -c "source activate mcass_kaz && \
    mkdir -p /opt/conda/envs/mcass_kaz/share/proj && \
    mkdir -p /opt/conda/envs/mcass_kaz/share/gdal && \
    # Use --no-dereference (-L) with cp if copying symlinks
    cp -Lr /usr/share/proj/* /opt/conda/envs/mcass_kaz/share/proj/ 2>/dev/null || true && \
    cp -Lr /usr/share/gdal/* /opt/conda/envs/mcass_kaz/share/gdal/ 2>/dev/null || true"

# Copy the rest of the application code into the final image
COPY . /app/

# Expose the port for the dashboard
EXPOSE 5007

# Set the default command to run the Panel dashboard
# Use bash -c to ensure source activate works
CMD ["bash", "-c", "source activate mcass_kaz && panel serve /app/dashboard/snowmapper.py --port 5007 --allow-websocket-origin=kaz.snowmapper.ch --allow-websocket-origin=0.0.0.0 --address 0.0.0.0"]