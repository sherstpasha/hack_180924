# Use the official PyTorch image with CUDA support (if you have a GPU)
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Alternatively, if you don't need GPU support, you can use the CPU image
# FROM pytorch/pytorch:1.12.1-cpu

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY . .

# Set environment variables (if any)
ENV PYTHONUNBUFFERED=1

# Set the entrypoint to your script
CMD ["python", "your_script.py"]
