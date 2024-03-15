ARG UBUNTU_VERSION=16.04
FROM --platform=linux/amd64 ubuntu:$UBUNTU_VERSION

# Install dependencies
RUN apt-get update \
  && apt-get install -y wget gcc make openssl libffi-dev libgdbm-dev libsqlite3-dev libssl-dev \
  zlib1g-dev git command-not-found less gzip vim x11-apps bzip2 libbz2-dev libfreetype6-dev \
  pkg-config libpng12-dev \
  && apt autoremove --yes \
  && apt-get clean

COPY . .

RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh.sh \
    && echo "export PATH="/root/miniforge3/bin:$PATH"" >> ~/.bashrc

ENV PATH /root/miniforge3/bin:$PATH
RUN conda init bash \
    && . ~/.bashrc \
    && mamba env create -y -f environment.yml \
    && conda activate CrystalPlan 

RUN python setup.py install

ENTRYPOINT [ "python crystalplan.py" ]
