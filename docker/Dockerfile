#########################################################
# Dockerfile pdf-content-extraction
# Install all system requirements into a docker container
# Authors:
#   Phillipe Cardenuto and Daniel Moreira
# Version Date: September 2020
##########################################################
FROM ubuntu:18.04
ENV LANG C.UTF-8

# Install system packages And PDFImages
RUN apt-get update && apt-get install -y \
    python3.6-dev \
    python3-pip\
    poppler-utils \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy solution to Docker
COPY PDF-Image-Extraction PDF-Image-Extraction
COPY print_page print_page
COPY facade facade

# Update pip3
RUN python3 -m pip install --upgrade pip

# Setup Unicamp solution
WORKDIR /work/PDF-Image-Extraction
RUN pip3 install -r requirements.txt

# Setup PageSegmenter solution
WORKDIR /work/print_page
RUN pip3 install -r requirements.txt

# Create input and output folders
RUN mkdir -p /INPUT && mkdir -p /OUTPUT

WORKDIR /work
COPY extract.sh .
RUN chmod +x extract.sh
CMD ["/work/extract.sh"]
