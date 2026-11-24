FROM python:3.11-slim

WORKDIR /app

# OpenCV headless needs a couple system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# default model dir inside the image
ENV MODEL_DIR=models/weights
ENV LOG_LEVEL=INFO
ENV PORT=5000

EXPOSE 5000

# production-ish entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "60", "app.main:app"]
