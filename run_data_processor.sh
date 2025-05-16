#!/bin/bash

# This script runs the data processor container to download and process the
# latest data from the SWE server running the snowmapperForecast model.
#
# Useage: nohup ./run_data_processor.sh &

# Set the path to your project directory
PROJECT_DIR=$(realpath ~/kaz_snowmapper_dashboard)

# Change to project directory
cd $PROJECT_DIR

# Get current timestamp for logging
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Create logs directory if it doesn't exist
mkdir -p $PROJECT_DIR/logs

# See if the data processor container is already running
if [ "$(docker ps -q -f name=kaz-snowmapper-processor-$(date +%Y%m%d))" ]; then
    echo "[$TIMESTAMP] Data processor is already running" >> $PROJECT_DIR/logs/processor.log
    exit 1
fi

# See if the image is already built
if [ "$(docker images -q mabesa/kaz-snowmapper-backend:latest)" ]; then
    echo "[$TIMESTAMP] Using existing image" >> $PROJECT_DIR/logs/processor.log
else
    echo "[$TIMESTAMP] Pulling image" >> $PROJECT_DIR/logs/processor.log
    docker pull mabesa/kaz-snowmapper-backend:latest 2>> $PROJECT_DIR/logs/processor.log
fi

# Run the data processor container
echo "[$TIMESTAMP] Starting data processor" 2>> $PROJECT_DIR/logs/processor.log

# Copy the following docker run command to the logs directory
echo "docker run command:" 2>> $PROJECT_DIR/logs/processor.log
echo "docker run --rm --name kaz-snowmapper-processor-$(date +%Y%m%d)" 2>> $PROJECT_DIR/logs/processor.log
echo "  --volume $PROJECT_DIR/data:/app/data" 2>> $PROJECT_DIR/logs/processor.log
echo "  --volume $PROJECT_DIR/logs:/app/logs" 2>> $PROJECT_DIR/logs/processor.log
echo "  --volume $PROJECT_DIR/processing/swe_server.pem:/app/processing/swe_server.pem:ro" 2>> $PROJECT_DIR/logs/processor.log
echo "  --env-file $PROJECT_DIR/.env mabesa/kaz-snowmapper-backend:latest" 2>> $PROJECT_DIR/logs/processor.log

# For debugging purposes, you can add the --rm flag to remove the container after it exits
docker run --rm \
  --name kaz-snowmapper-processor-$(date +%Y%m%d) \
  --volume $PROJECT_DIR/data:/app/data \
  --volume $PROJECT_DIR/logs:/app/logs \
  --volume $PROJECT_DIR/processing/swe_server.pem:/app/processing/swe_server.pem:ro \
  --env-file $PROJECT_DIR/.env \
  mabesa/kaz-snowmapper-backend:latest 2>> $PROJECT_DIR/logs/processor.log

EXIT_CODE=$?

# Log the completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$TIMESTAMP] Data processor completed successfully" 2>> $PROJECT_DIR/logs/processor.log
else
    echo "[$TIMESTAMP] Data processor failed with exit code $EXIT_CODE" 2>> $PROJECT_DIR/logs/processor.log
fi