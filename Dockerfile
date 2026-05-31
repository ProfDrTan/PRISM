FROM python:3.10-slim

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DEMO_MODE=true \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    DISABLE_SAFETENSORS_CONVERSION=1

WORKDIR $HOME/app

# Install dependencies first (Docker layer cache optimisation)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . $HOME/app

# Create writable data directories
RUN mkdir -p $HOME/app/data/demo_guests \
             $HOME/app/data/LLM_reports \
             $HOME/app/data/sessions \
             $HOME/app/data/charts \
             $HOME/app/runtime_data \
    && chmod -R 777 $HOME/app/data \
    && chmod -R 777 $HOME/app/runtime_data

EXPOSE 7860

# Production WSGI server
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
