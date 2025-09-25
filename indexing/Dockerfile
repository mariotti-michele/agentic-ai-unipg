FROM python:3.12-slim

# Installa dipendenze di sistema richieste da camelot, unstructured e pi-heif
RUN apt-get update && apt-get install -y \
    build-essential \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    libheif-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Imposta directory di lavoro
WORKDIR /app

# Copia requirements.txt e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice
COPY . .

# Mantieni il container vivo (puoi sostituirlo con il comando che vuoi eseguire di default)
CMD ["tail", "-f", "/dev/null"]
