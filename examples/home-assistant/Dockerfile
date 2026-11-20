ARG SCYLLASBAND_COMMIT=ece0adcce0896a77d6ff11adc2e3d86f56b3a51e
ARG WYOMING_SPEC=">=1.7,<2"

FROM python:3.12-slim

ARG SCYLLASBAND_COMMIT
ARG WYOMING_SPEC

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app

RUN pip install --no-cache-dir \
      "https://github.com/lowkeytea/scyllasband/archive/${SCYLLASBAND_COMMIT}.tar.gz" \
      "wyoming${WYOMING_SPEC}"

COPY server.py /app/server.py

# Bake the LiteRT bundle (~470 MB) into the image so runtime needs no network.
RUN python -m scyllasband download --flavor litert --models-dir /models -y \
 && useradd --create-home --uid 10001 scylla \
 && chown -R scylla:scylla /app /models

USER scylla

EXPOSE 10210

CMD ["python", "/app/server.py"]
