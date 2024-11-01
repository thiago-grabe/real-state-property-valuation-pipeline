# Use an appropriate base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app/src"

# Expose the port the API will run on
EXPOSE 8000
