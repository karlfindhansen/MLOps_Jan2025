
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*


RUN mkdir /app



WORKDIR /app

COPY src/requirements_backend.txt /app/requirements_backend.txt
COPY src/backend.py /app/backend.py
COPY src src
COPY pyproject.toml pyproject.toml
COPY requirements.txt requirements.txt



RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend.txt


EXPOSE $PORT
CMD exec uvicorn --port $PORT --host 0.0.0.0 backend:app