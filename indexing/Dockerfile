FROM python:3.12-slim

# Install dipendenze di sistema di base (per Camelot, OCR e PDF)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ghostscript \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Imposta directory di lavoro
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installa dipendenze Playwright + Chromium
RUN playwright install-deps
RUN playwright install chromium

# Copia il codice
COPY . .

CMD ["tail", "-f", "/dev/null"]