FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install JupyterLab
RUN pip install --no-cache-dir jupyterlab

# Set JupyterLab as default interface
ENV JUPYTER_ENABLE_LAB=yes

# Copy project files
COPY . .

# Create necessary directories for datamart
RUN mkdir -p datamart/bronze datamart/silver datamart/gold

# Expose JupyterLab port
EXPOSE 8888

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.allow_origin='*'"]