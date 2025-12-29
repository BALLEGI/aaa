FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système (nécessaire pour watchdog parfois)
RUN apt-get update && apt-get install -y gcc

# Copie des requirements et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie des scripts
COPY *.py .

# Point d'entrée par défaut (sera override dans le compose)
CMD ["python", "-u"]
