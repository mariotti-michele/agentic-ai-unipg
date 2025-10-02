FROM python:3.12-slim

# Install dipendenze di sistema di base (per Camelot, OCR e PDF)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ghostscript \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-ita \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Aggiorna pip, setuptools e wheel
RUN pip install --upgrade pip setuptools wheel

# Imposta directory di lavoro
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --default-timeout=10000 --no-cache-dir -r requirements.txt

# Installa dipendenze Playwright + Chromium
RUN playwright install-deps
RUN playwright install chromium

# Copia il codice
COPY . .

CMD ["tail", "-f", "/dev/null"]