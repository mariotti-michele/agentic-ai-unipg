FROM python:3.12-slim

# Install dipendenze di sistema (opzionali, se servono tesseract o playwright aggiungerli qui)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice
COPY . .

# Comando di default
CMD ["python", "main.py"]
