FROM ubuntu:latest
RUN mkdir /app
VOLUME /out
ADD . /app
WORKDIR /app
RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN cd /usr/local/bin && ln -s /usr/bin/python3 python
RUN pip3 --no-cache-dir install --upgrade pip
RUN chmod +x *.sh
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python"]