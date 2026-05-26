# Usamos la imagen oficial ligera de Python
FROM python:3.14

# Instalar uv directamente desde los binarios oficiales
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Instalar las dependencias del sistema necesarias para compilar paquetes de C (como psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar los archivos de dependencias primero (optimiza el cache de Docker)
COPY pyproject.toml uv.lock ./

# Instalar todas las dependencias usando uv
RUN uv sync --frozen --no-cache

# Copiar el resto del código del proyecto
COPY . .

# Por defecto, correr el demonio del scheduler
CMD ["uv", "run", "scheduler.py"]