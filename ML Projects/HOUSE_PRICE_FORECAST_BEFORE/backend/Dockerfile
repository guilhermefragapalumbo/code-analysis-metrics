FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3.9 python3-pip tzdata

RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN pip3 install --upgrade pip && \
    pip3 install setuptools wheel

# set working directory
ADD . /app
WORKDIR /app

RUN pip3 install flask flask_cors joblib pandas xgboost numpy

EXPOSE 8081

CMD python3 app.py
