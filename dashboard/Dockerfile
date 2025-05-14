# Use a lightweight base image with Python and Conda
FROM continuumio/miniconda3:4.12.0 AS builder

# Set environment variables for Conda
ENV PATH="/opt/conda/bin:$PATH"

# Create a Conda environment and install GDAL 3.10 and other dependencies
COPY requirements.yml /tmp/requirements.yml
RUN conda env create -f /tmp/requirements.yml && \
    conda clean --all --yes

# Activate the Conda environment
ENV PATH="/opt/conda/envs/mcass_kaz/bin:$PATH"
ENV CONDA_DEFAULT_ENV=mcass_kaz

# Install additional Python dependencies via pip
COPY pip_requirements.txt /tmp/pip_requirements.txt
RUN pip install --no-cache-dir -r /tmp/pip_requirements.txt

# Final stage
FROM continuumio/miniconda3:4.12.0

# Copy Conda environment from the builder stage
COPY --from=builder /opt/conda /opt/conda
ENV PATH="/opt/conda/envs/mcass_kaz/bin:$PATH"
ENV CONDA_DEFAULT_ENV=mcass_kaz

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 dashboard

# Create logs directory and set proper permissions
RUN mkdir -p /app/logs && \
    chown -R dashboard:dashboard /app && \
    chmod 755 /app && \
    chmod 777 /app/logs

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=dashboard:dashboard . /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DASHBOARD_ENV=aws \
    PORT=5006

# Switch to non-root user
USER dashboard

# Expose port
EXPOSE ${PORT}

# Run the panel server
CMD ["sh", "-c", "DASHBOARD_ENV=aws panel serve /app/dashboard/snowmapper.py --port $PORT --allow-websocket-origin=kaz.snowmapper.ch --address 0.0.0.0"]