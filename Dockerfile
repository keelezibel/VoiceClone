# Python 3.6
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ARG PYTHON_VERSION=3.7
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates gcc libsndfile1 libportaudio2 python3-dev && \
  wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
  echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
  /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
  rm -rf /tmp/* && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN conda install -y --quiet python=$PYTHON_VERSION

# Packages that we need
COPY requirements.txt .
WORKDIR .

RUN  pip install --upgrade pip && \
  pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org  -r requirements.txt 

COPY . .

EXPOSE 8000

# waitress-serve --listen 0.0.0.0:8000 synth_webservice:__hug_wsgi__
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8000", "synth_webservice:__hug_wsgi__"]
#CMD ["waitress-serve --listen 0.0.0.0:8000 synth_webservice:__hug_wsgi__"]
