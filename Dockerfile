FROM python:3.10-slim

# Create a non-root user for security
RUN useradd -m -u 1000 user
WORKDIR /app

# Install CPU-only PyTorch and requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, templates, and trained ResNet-18 weights
COPY --chown=user app.py .
COPY --chown=user src/ ./src/
COPY --chown=user templates/ ./templates/
COPY --chown=user models/resnet18_best.pth ./models/resnet18_best.pth

# Create writable directories for SQLite telemetry and Active Learning vaults
RUN mkdir -p data/feedback_vault data/quarantine_vault models && \
    chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

EXPOSE 7860

# Run with 1 worker and 2 threads to stay well within Render's 512MB RAM limit
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]