FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the metadata needed for `pip install .` (pyproject.toml references README.md).
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .

# Copy the runtime source. scripts/ holds the reference-corpus ingester;
# data/ holds the seed reference corpus; db/ holds the SQL schema.
COPY app ./app
COPY db ./db
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
