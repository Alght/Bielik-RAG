FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

RUN mkdir -p /data
RUN mkdir -p /logs
RUN mkdir -p /output

RUN mkdir -p /original_documents
RUN mkdir -p /text_documents
