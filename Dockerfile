# Production Dockerfile for GLUCOM Explainable Glaucoma Detection Platform
FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV MPLCONFIGDIR=/app/.matplotlib_cache
ENV PORT=5000

# Install system dependencies required by OpenCV and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart

# Copy complete project codebase
COPY . /app/

# Ensure runtime directories exist
RUN mkdir -p /app/data /app/checkpoints /app/results /app/uploads /app/static/generated /app/.matplotlib_cache

# Create non-root system user for security
RUN useradd -m -u 1001 glucom && \
    chown -R glucom:glucom /app

USER glucom

# Expose production port (dynamically overridden by cloud providers)
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Production entrypoint supporting dynamic cloud $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-5000}"]

