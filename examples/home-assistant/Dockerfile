FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install "git+https://github.com/lowkeytea/scyllasband.git" wyoming

COPY server.py /app/server.py

# Bake the LiteRT bundle (~470 MB) into the image so runtime needs no network.
RUN python -m scyllasband download --flavor litert --models-dir /models -y

EXPOSE 10210

CMD ["python", "/app/server.py"]
