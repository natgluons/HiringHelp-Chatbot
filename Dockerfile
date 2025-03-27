# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install nginx
RUN apt-get update && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Configure pip and install requirements
RUN pip config set global.index-url https://pypi.org/simple && \
    pip config set global.timeout 1000 && \
    pip install --no-cache-dir --retries 3 --timeout 1000 -r requirements.txt

# Copy the application files
COPY . .

# Copy nginx configuration
RUN echo 'server { \
    listen 7860; \
    server_name _; \
    root /app; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    location /api { \
        proxy_pass http://127.0.0.1:5000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
}' > /etc/nginx/sites-available/default

# Make port 7860 available (Hugging Face Spaces uses this port)
EXPOSE 7860

# Create startup script
RUN echo '#!/bin/bash\n\
nginx\n\
FLASK_APP=backend/app.py flask run --host=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run both nginx and Flask when the container launches
CMD ["/app/start.sh"]
