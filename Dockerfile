FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend backend
COPY frontend frontend
RUN useradd -m studio && mkdir -p /app/data && chown -R studio:studio /app
USER studio
EXPOSE 4173
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:4173/health')"
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "4173"]
