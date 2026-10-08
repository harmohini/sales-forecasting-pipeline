FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for PostgreSQL and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and model artifacts
COPY . .

# Expose API server port
EXPOSE 8000

# Production launch command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

