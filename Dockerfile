FROM python:3.11.6-slim-bookworm

COPY src/ql_toolkit/ /usr/bin/ql_toolkit
COPY src/elasticity/ /usr/bin/elasticity
COPY src/report/ /usr/bin/report
COPY src/requirements.txt /usr/bin/elasticity/
COPY src/main.py /usr/bin/train

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y build-essential libblas-dev gcc && \
    rm -rf /var/lib/apt/lists/* &&\
    apt-get clean


RUN pip3 install --no-cache-dir -U pip setuptools wheel && \
    pip3 install --no-cache-dir -r /usr/bin/elasticity/requirements.txt

# Make the main files executables
RUN chmod 755 /usr/bin/train