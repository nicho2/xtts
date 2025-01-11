# -----------------------------------------------------------------------------
# 1) Image de base
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Éviter les interactions lors des installations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# -----------------------------------------------------------------------------
# 2) Installation des dépendances de base + ajout du PPA "deadsnakes"
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update

# -----------------------------------------------------------------------------
# 3) Installation de Python 3.10 + venv + distutils + dev
# -----------------------------------------------------------------------------
RUN apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Paris /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 4) (Optionnel) Faire de Python 3.10 la version par défaut
# -----------------------------------------------------------------------------
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# -----------------------------------------------------------------------------
# 5) Définir le répertoire de travail
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# 6) Copier vos sources
# -----------------------------------------------------------------------------
COPY examples /app/examples
COPY README.md /app/README.md
COPY app.py /app/
COPY start.sh /app/
COPY requirements.txt /app/requirements.txt

# -----------------------------------------------------------------------------
# 7) Mettre pip à jour et créer le venv
# -----------------------------------------------------------------------------
RUN python3.10 -m ensurepip --upgrade
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m venv /app/venv

# -----------------------------------------------------------------------------
# 8) Installer vos dépendances dans le venv
# -----------------------------------------------------------------------------
RUN /bin/bash -c "source /app/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt \
    && pip install TTS \
    "

# -----------------------------------------------------------------------------
# 9) Exposer le port (si nécessaire)
# -----------------------------------------------------------------------------
EXPOSE 7860

# -----------------------------------------------------------------------------
# 10) Commande de démarrage
# -----------------------------------------------------------------------------
CMD ["bash", "start.sh"]
