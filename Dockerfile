FROM mcr.microsoft.com/devcontainers/python:3.11

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

