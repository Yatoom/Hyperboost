FROM ubuntu:22.04
WORKDIR /app
COPY ../requirements.txt .
RUN apt update -y && apt install pip -y
RUN pip install -r /app/requirements.txt
ENV PYTHONPATH=/app