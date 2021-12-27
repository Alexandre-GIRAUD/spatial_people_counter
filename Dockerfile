# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

# Install pip requirements
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
