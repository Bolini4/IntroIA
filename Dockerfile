FROM ubuntu:22.04

RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get install -y python3
RUN apt-get install -y python3-pip

RUN pip3 install tensorflow
RUN pip3 install keras