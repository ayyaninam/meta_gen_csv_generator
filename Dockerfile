# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for input and output
RUN mkdir -p /app/input /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
if [ -z "$NUM_SUBTOPICS" ] || [ -z "$NUM_SUBSUBTOPICS" ]; then\n\
    echo "Please provide NUM_SUBTOPICS and NUM_SUBSUBTOPICS environment variables"\n\
    exit 1\n\
fi\n\
cp $INPUT_DIR/* /app/\n\
python generate_topics.py $NUM_SUBTOPICS $NUM_SUBSUBTOPICS\n\
cp Generated_Topics.csv $OUTPUT_DIR/\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 