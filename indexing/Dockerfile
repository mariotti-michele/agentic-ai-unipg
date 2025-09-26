FROM python:3.12-slim

<<<<<<< HEAD
# Install dipendenze di sistema di base (per Camelot, OCR e PDF)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ghostscript \
    poppler-utils \
    tesseract-ocr \
    curl \
=======
# Installa dipendenze di sistema richieste da camelot, unstructured e pi-heif
RUN apt-get update && apt-get install -y \
    build-essential \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    libheif-dev \
    poppler-utils \
    tesseract-ocr \
>>>>>>> 6f2d101314c11436a06ba1b487f388a9a016bf14
    && rm -rf /var/lib/apt/lists/*

# Imposta directory di lavoro
WORKDIR /app

<<<<<<< HEAD
# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installa dipendenze Playwright + Chromium
RUN playwright install-deps
RUN playwright install chromium

# Copia il codice
COPY . .

CMD ["tail", "-f", "/dev/null"]
=======
# Copia requirements.txt e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice
COPY . .

# Mantieni il container vivo (puoi sostituirlo con il comando che vuoi eseguire di default)
CMD ["tail", "-f", "/dev/null"]
>>>>>>> 6f2d101314c11436a06ba1b487f388a9a016bf14
