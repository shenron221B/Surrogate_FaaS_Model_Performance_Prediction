FROM python:3.9-slim

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    numpy==1.24.3

# Download MobileNet-SSD model and config
RUN wget -q https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt -O /deploy.prototxt && \
    wget -q https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel -O /mobilenet_ssd.caffemodel

# Copy detection script
COPY executor.py /
COPY function.py /

# Set environment variable to avoid OpenCV GUI warnings
ENV DISPLAY=:0

# Set working directory
WORKDIR /

CMD ["python", "executor.py"]