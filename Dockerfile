# This is a Dockerfile for setting up an Ubuntu box with pystripe installed

FROM continuumio/anaconda3

RUN conda install numpy==1.14

RUN apt-get install -y git
RUN git clone https://github.com/chunglabmit/pystripe.git
RUN cd /pystripe; python setup.py install
