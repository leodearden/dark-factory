FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY config/ config/
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "fused_memory.server.main"]
