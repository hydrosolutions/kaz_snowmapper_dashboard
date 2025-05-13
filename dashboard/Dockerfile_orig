# Use multi-stage build for smaller final image
FROM windpioneers/gdal-python:tart-dragonfly-gdal-3.9.3-python-3.11-slim AS builder
#FROM python:3.11-slim AS builder

# Install build dependencies including GDAL
#RUN apt-get update && apt-get install -y \
#    curl \
#    build-essential \
#    libgeos-dev \
#    gdal-bin\
#    libgdal-dev \
#    libproj-dev \
#    proj-bin \
#    && rm -rf /var/lib/apt/lists/*

# Install PROJ 9.3.1 from source
#RUN wget https://download.osgeo.org/proj/proj-9.3.1.tar.gz && \
#    tar -xzf proj-9.3.1.tar.gz && \
#    cd proj-9.3.1 && \
#    ./autogen.sh && \
#    ./configure && \
#    make && \
#    make install && \
#    cd .. && \
#    rm -rf proj-9.3.1 proj-9.3.1.tar.gz

# Install GDAL 3.6.2 from source
#RUN wget https://download.osgeo.org/gdal/3.6.2/gdal-3.6.2.tar.gz && \
#    tar -xzf gdal-3.6.2.tar.gz && \
#    cd gdal-3.6.2 && \
#    ./configure && \
#    make && \
#    make install && \
#    cd .. && \
#    rm -rf gdal-3.6.2 gdal-3.6.2.tar.gz

# Set GDAL environment variables for building
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_DATA=/usr/share/proj
ENV PATH="/usr/bin:${PATH}"

# Get GDAL version and set as env variable
RUN export GDAL_VERSION=$(gdal-config --version) && \
    echo "GDAL_VERSION=${GDAL_VERSION}" && \
    export GDAL_VERSION

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install remaining dependencies
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    #wget \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set GDAL environment variables for runtime
#ENV GDAL_CONFIG=/usr/bin/gdal-config
#ENV PROJ_LIB=/usr/share/proj

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
# Comment line above and uncomment line below if you want to test the dashboard without the websocket origin restriction
#CMD ["sh", "-c", "DASHBOARD_ENV=aws panel serve /app/dashboard/snowmapper.py --port $PORT --allow-websocket-origin=<IP-Address of your server> --address 0.0.0.0"]
