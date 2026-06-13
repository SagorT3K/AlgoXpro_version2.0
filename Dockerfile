FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt /app/backend/requirements.txt
COPY pyquotex/pyproject.toml /app/pyquotex/pyproject.toml

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] websockets httpx \
    python-dotenv pandas numpy ta redis async-timeout && \
    pip install --no-cache-dir -e /app/pyquotex 2>/dev/null || true

# Copy application code
COPY backend/ /app/backend/
COPY pyquotex/ /app/pyquotex/
COPY frontend/ /app/frontend/

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8000

# Run with multiple workers
CMD ["python", "-m", "uvicorn", "backend.main_scaled:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--limit-concurrency", "1000", \
     "--timeout-keep-alive", "65"]
