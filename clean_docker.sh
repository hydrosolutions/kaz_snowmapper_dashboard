#!/bin/bash

# This script cleans up the docker containers and images used by the data processor

# Set the path to your project directory
PROJECT_DIR=~/kaz_snowmapper_dashboard

# Change to project directory
cd $PROJECT_DIR

# Get current timestamp for logging
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Stop and remove all data processor containers
echo "[$TIMESTAMP] Stopping and removing data processor containers" >> $PROJECT_DIR/logs/clean.log
docker stop $(docker ps -q -f name=kaz-snowmapper-processor) >> $PROJECT_DIR/logs/clean.log 2>&1
docker rm $(docker ps -aq -f name=kaz-snowmapper-processor) >> $PROJECT_DIR/logs/clean.log 2>&1

# Remove the data processor image
echo "[$TIMESTAMP] Removing data processor image" >> $PROJECT_DIR/logs/clean.log
docker rmi mabesa/kaz-snowmapper-backend:latest >> $PROJECT_DIR/logs/clean.log 2>&1

# Log the completion status
echo "[$TIMESTAMP] Clean up completed" >> $PROJECT_DIR/logs/clean.log


