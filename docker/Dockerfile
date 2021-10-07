FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo /home/user/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && chmod +x /home/user/miniconda.sh \
    && /home/user/miniconda.sh -b -p /home/user/miniconda \
    && rm /home/user/miniconda.sh \
    && conda install -y python==3.6.9 \
    && conda clean -ya



# Install Jupyter Notebook
RUN conda install -y -c conda-forge jupyterlab

# selecting a work dir also creates the directory, so we use it to create /data as a mounting point
WORKDIR /data

# our actual working directory
WORKDIR /workingdir

#make port 8888 available from outside, just in case the container is supposed to run jupyter notebooks
EXPOSE 8888

#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
#ADD requirements.txt .
#RUN python -m pip install -r requirements.txt
WORKDIR /.cache
RUN chmod 777 /.cache

WORKDIR /.config/matplotlib
RUN chmod 777 /.config/matplotlib

WORKDIR /workingdir
WORKDIR /workingdir/data
RUN chmod 777 /workingdir/data
WORKDIR /workingdir

# Install requirements
COPY requirements.txt /workingdir/requirements.txt
RUN pip install -r requirements.txt
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

# update pythonpath
ENV PYTHONPATH "${PYTHONPATH}:/workingdir"

# matplotlib config (used by benchmark)
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
