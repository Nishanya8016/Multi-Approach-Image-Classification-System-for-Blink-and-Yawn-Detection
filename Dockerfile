# Dockerfile
# Use a robust base image with Python
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PORT 7860  # Hugging Face Spaces default port

# Install system dependencies needed by OpenCV (this is critical!)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy files
COPY requirements.txt /app/
COPY app.py /app/
COPY vgg16_transfer_model.h5 /app/
COPY templates/ /app/templates/

# Install Python dependencies
RUN pip install -r requirements.txt

# Run the application using Gunicorn (a production web server)
# Gunicorn uses 'app:app' where the first 'app' is the filename (app.py) 
# and the second 'app' is the Flask application instance variable (app = Flask(__name__))
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 app:app