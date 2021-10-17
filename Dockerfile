#torch==1.6.0
#torchvision==0.7.0
#python==3.7.7
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
SHELL ["/bin/bash", "-c"]


# Update
RUN apt -y update
RUN apt-get install -y apt-utils
RUN apt -y upgrade
RUN apt-get install -y build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev git
RUN pip install --upgrade pip

# Copy the code
COPY ./src /src
COPY ./scripts /scripts
WORKDIR /

RUN chmod 775 /scripts/01_cmfd_all_images.py
RUN chmod 775 /scripts/02_eval_all_results.py

# Instal the needed python libraries.
RUN pip install -r /src/requirements.txt

# Instal CRAFT: Character-Region Awareness For Text detection
RUN cd /src && bash download_CRAFT.sh

# Create the input-output folder
ENV CMFD_IO=/cmfd/io
RUN mkdir -p /cmfd/io

WORKDIR /