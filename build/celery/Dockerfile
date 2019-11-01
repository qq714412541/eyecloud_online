FROM tensorflow/tensorflow:1.12.0-gpu-py3

WORKDIR /app

COPY requirements.txt /app

RUN pip install -i https://pypi.douban.com/simple/ -r requirements.txt

RUN adduser --disabled-password --gecos '' myuser