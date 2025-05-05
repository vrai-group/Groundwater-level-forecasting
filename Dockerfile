FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3-pip

# Update symlink to point to latest
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
RUN python3 --version
RUN pip3 --version

#System and libraries
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5dbus5 \
    qttools5-dev \
    qttools5-dev-tools \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libxfixes-dev \
    libx11-xcb-dev \
    libxcb-glx0-dev \
	git

RUN python3 --version
RUN pip3 --version
RUN pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install	torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

#environment
ENV APPDIR gwl
WORKDIR gwl

RUN cd /gwl/

CMD ["bash"]

