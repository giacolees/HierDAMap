# Use NVIDIA CUDA 11.8 base image (latest LTS release with broad compatibility)
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# Copy requirements first to leverage Docker cache
COPY requirements_f.txt .

# Install Python dependencies
RUN pip install -r requirements_f.txt

# Copy the rest of the application
COPY . .

