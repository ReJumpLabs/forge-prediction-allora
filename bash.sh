#!/bin/bash

# Number of workers to create
total_workers=450

# Create the Docker Compose file
compose_file="docker-compose.yml"

# Start writing the compose file
cat <<EOF > $compose_file
version: '3.8'

services:
EOF

# Add the single inference service
cat <<EOF >> $compose_file
  inference:
    container_name: inference
    build:
      context: .
      dockerfile: Dockerfile
    command: python -u /app/app.py
    ports:
      - "8000:8000"
EOF

# Generate workers for the single inference service
for worker in $(seq 1 $total_workers); do
  cat <<EOF >> $compose_file
  worker$worker:
    container_name: worker$worker
    image: alloranetwork/allora-offchain-node:latest
    volumes:
      - ./worker-data:/data
    depends_on:
      - inference
    env_file:
      - ./worker-data/env_file_$worker
EOF
done

# Add volumes section at the end
cat <<EOF >> $compose_file

volumes:
  inference-data:
  worker-data:
EOF

echo "Docker Compose file generated successfully as $compose_file"
