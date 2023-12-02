##############
# Base Image #
##############

FROM rtvu/docker-ubuntu

################
# Set up conda #
################

RUN \
  mkdir ${HOME}/miniconda3 && \
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ${HOME}/miniconda3/miniconda.sh && \
  bash ${HOME}/miniconda3/miniconda.sh -b -u -p ${HOME}/miniconda3 && \
  rm -rf ${HOME}/miniconda3/miniconda.sh && \
  ${HOME}/miniconda3/bin/conda init bash

######################
# Set up environment #
######################

COPY --chown=<user>:<user> environment.yaml .

RUN \
  ${HOME}/miniconda3/bin/conda env create -f environment.yaml && \
  rm environment.yaml && \
  echo "" >> ${HOME}/.bashrc && \
  echo "conda activate deep_momentum_networks" >> ${HOME}/.bashrc

#####################
# Configure startup #
#####################

CMD ["/bin/bash"]
