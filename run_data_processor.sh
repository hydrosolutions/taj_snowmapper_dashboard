#!/bin/bash

# Set the path to your project directory
PROJECT_DIR=~/taj_snowmapper_dashboard

# Change to project directory
cd $PROJECT_DIR

# Get current timestamp for logging
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Run the data processor container
echo "[$TIMESTAMP] Starting data processor" >> $PROJECT_DIR/logs/processor.log

docker run --rm \
  --name taj-snowmapper-processor-$(date +%Y%m%d) \
  --volume $PROJECT_DIR/data:/app/data \
  --volume $PROJECT_DIR/logs:/app/logs \
  --volume $PROJECT_DIR/processing/swe_server.pem:/app/processing/swe_server.pem:ro \
  --env-file $PROJECT_DIR/.env \
  taj-snowmapper-backend:latest 2>> $PROJECT_DIR/logs/processor.log

EXIT_CODE=$?

# Log the completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$TIMESTAMP] Data processor completed successfully" >> $PROJECT_DIR/logs/processor.log
else
    echo "[$TIMESTAMP] Data processor failed with exit code $EXIT_CODE" >> $PROJECT_DIR/logs/processor.log
fi