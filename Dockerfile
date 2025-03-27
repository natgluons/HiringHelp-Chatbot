# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Configure pip to use a more reliable mirror and increase timeout
RUN pip config set global.index-url https://pypi.org/simple && \
    pip config set global.timeout 1000 && \
    pip install --no-cache-dir --retries 3 --timeout 1000 -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=backend/app.py

# Run flask command when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
