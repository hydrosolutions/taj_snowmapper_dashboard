# Use multi-stage build for smaller final image
FROM python:3.11-slim AS builder

# Install build dependencies including GDAL
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables for building
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV PROJ_LIB=/usr/share/proj
ENV PATH="/usr/bin:${PATH}"

# Get GDAL version and set as env variable
RUN export GDAL_VERSION=$(gdal-config --version) && \
    echo "GDAL_VERSION=${GDAL_VERSION}" && \
    export GDAL_VERSION

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install GDAL Python bindings first with explicit version
RUN GDAL_VERSION=$(gdal-config --version) && \
    pip install --no-cache-dir GDAL==${GDAL_VERSION}

# Copy requirements and install remaining dependencies
COPY dashboard/requirements.txt .
RUN GDAL_VERSION=$(gdal-config --version) && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set GDAL environment variables for runtime
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV PROJ_LIB=/usr/share/proj

# Create non-root user
RUN useradd -m -u 1000 dashboard

# Create logs directory and set proper permissions
RUN mkdir -p /app/logs && \
    chown -R dashboard:dashboard /app && \
    chmod 755 /app && \
    chmod 777 /app/logs

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=dashboard:dashboard . /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DASHBOARD_ENV=aws \
    PORT=${PORT}

# Switch to non-root user
USER dashboard

# Expose port
EXPOSE ${PORT}

# Run the panel server
CMD ["sh", "-c", "panel serve /app/dashboard/snowmapper.py --port $PORT --allow-websocket-origin=taj.snowmapper.ch --address 0.0.0.0"]
#CMD panel serve /app/dashboard/snowmapper.py --port ${PORT}

