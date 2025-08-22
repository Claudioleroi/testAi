# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers requis
COPY requirements.txt .
COPY main.py .

# Créer le dossier pour les modèles
RUN mkdir -p models

# Copier le modèle GGUF (à adapter selon votre structure)

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]