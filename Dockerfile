# ── Customer Support Triage — OpenEnv Docker Image ───────────────────────────
# Deploys as a Hugging Face Space (SDK: docker, port: 7860)
# Build:  docker build -t openenv-support-triage .
# Run:    docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... openenv-support-triage

FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-hackathon"
LABEL org.opencontainers.image.title="Customer Support Triage OpenEnv"
LABEL org.opencontainers.image.description="OpenEnv environment for customer support ticket triage"
LABEL org.opencontainers.image.version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port (HF Spaces requires 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
