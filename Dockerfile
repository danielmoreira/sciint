# Basis image: Ubuntu 18.04
FROM ubuntu:18.04

# Installs the needed system packages.
RUN apt-get update
RUN apt-get install -y python3.6-dev python3-pip
RUN apt-get install -y tesseract-ocr
RUN apt-get install -y libsm6

# Copies the solution to docker.
COPY ./ /provenance
WORKDIR /provenance
RUN chmod 775 /provenance/01_build_all_graphs.sh
RUN chmod 775 /provenance/02_eval_all_graphs.sh

# Creates the input-output folder
# (it must be mounted as a volume when executing this container).
RUN mkdir -p /provenance/io

# Installs the needed python libraries.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
