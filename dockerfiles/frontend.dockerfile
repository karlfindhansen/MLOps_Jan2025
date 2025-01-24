
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*


RUN mkdir /app



WORKDIR /app

COPY src/requirements_frontend.txt /app/requirements_frontend.txt
COPY src/frontend.py /app/frontend.py
COPY src/explainability.py /app/explainability.py
COPY src/config /app/config
COPY pyproject.toml pyproject.toml
COPY requirements.txt requirements.txt



RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt


EXPOSE $PORT

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port", "$PORT", "--server.address=0.0.0.0"]
