FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Add Display to the container. This is needed for the dashboard to work.
ENV DISPLAY=:0

# Set environment variable for Python to write output unbuffered
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data

# Expose the port that the application listens on.
EXPOSE 5006

# Run the application
CMD ["panel", "serve", "snowmapper.py", "--port", "5006", "--allow-websocket-origin", "localhost:5006"]