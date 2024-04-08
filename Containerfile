# Dockerfile for base image of all pangeo images
FROM ubuntu:22.04
# build file for pangeo images

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Setup environment to match variables set by repo2docker as much as possible
# The name of the conda environment into which the requested packages are installed
ENV CONDA_ENV=cisl-gpu-base \
    # Tell apt-get to not block installs by asking for interactive human input
    DEBIAN_FRONTEND=noninteractive \
    # Set username, uid and gid (same as uid) of non-root user the container will be run as
    NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=1000 \
    # Use /bin/bash as shell, not the default /bin/sh (arrow keys, etc don't work then)
    SHELL=/bin/bash \
    # Setup locale to be UTF-8, avoiding gnarly hard to debug encoding errors
    LANG=C.UTF-8  \
    LC_ALL=C.UTF-8 \
    # Install conda in the same place repo2docker does
    CONDA_DIR=/srv/conda \
    CONDA_USR_DIR=/srv/base-conda \
    PIP_EXTRA_INDEX_URL='https://pypi.nvidia.com'

# All env vars that reference other env vars need to be in their own ENV block
# Path to the python environment where the jupyter notebook packages are installed
ENV NB_PYTHON_PREFIX=${CONDA_USR_DIR}/envs/${CONDA_ENV} \
    # Home directory of our non-root user
    HOME=/home/${NB_USER}

# Add both our notebook env as well as default conda installation to $PATH
# Thus, when we start a `python` process (for kernels, or notebooks, etc),
# it loads the python in the notebook conda environment, as that comes
# first here.
ENV PATH=${CONDA_DIR}/bin:${NB_PYTHON_PREFIX}/bin:${PATH}

# Ask dask to read config from ${CONDA_DIR}/etc rather than
# the default of /etc, since the non-root jovyan user can write
# to ${CONDA_DIR}/etc but not to /etc
ENV DASK_ROOT_CONFIG=${CONDA_USR_DIR}/etc

COPY packages/apt.txt apt.txt

# Install apt packages
RUN echo "Installing Apt-get packages..." \
    && apt-get update --fix-missing > /dev/null \
    && apt-get install -y apt-utils wget zip tzdata > /dev/null \
    && xargs -a apt.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
###
# Copy a script that we will use to correct permissions after running certain commands
###
COPY scripts/fix-permissions /usr/local/bin/fix-permissions

RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
# hadolint ignore=SC2016
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc
    # More information in: https://github.com/jupyter/docker-stacks/pull/2047
    # and docs: https://docs.conda.io/projects/conda/en/latest/dev-guide/deep-dives/activation.html

# Create NB_USER with name jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN echo "Creating ${NB_USER} user..." \
    # Create a group for the user to be part of, with gid same as uid
    && groupadd --gid ${NB_UID} ${NB_USER}  \
    # Create non-root user, with given gid, uid and create $HOME
    && useradd --create-home --gid ${NB_UID} --no-log-init --uid ${NB_UID} ${NB_USER} \
    # Make sure that /srv is owned by non-root user, so we can install things there
    && chown -R ${NB_USER}:${NB_USER} /srv

RUN mkdir -p "${CONDA_DIR}" && \
    mkdir -p "${CONDA_USR_DIR}" && \
    #chown "${NB_USER}:${NB_UID}" "${CONDA_DIR}" && \
    #chown "${NB_USER}:${NB_UID}" "${CONDA_USR_DIR}" && \
    #chmod g+w /etc/passwd && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "${CONDA_USR_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Add TZ configuration - https://github.com/PrefectHQ/prefect/issues/3061
ENV TZ UTC
# ========================

# Install latest mambaforge in ${CONDA_DIR}
RUN echo "Installing Mambaforge..." \
    && URL="https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh" \
    && wget --quiet ${URL} -O installer.sh \
    && /bin/bash installer.sh -u -b -p ${CONDA_DIR} \
    && rm installer.sh \
    && mamba install conda-lock jupyterhub==4.0.1 notebook conda-forge::nb_conda_kernels -y \
    && mamba clean -afy \
    # After installing the packages, we cleanup some unnecessary files
    # to try reduce image size - see https://jcristharif.com/conda-docker-tips.html
    # Although we explicitly do *not* delete .pyc files, as that seems to slow down startup
    # quite a bit unfortunately - see https://github.com/2i2c-org/infrastructure/issues/2047
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    # Fix permissions
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "${CONDA_USR_DIR}" \
    && fix-permissions "/home/${NB_USER}"

WORKDIR /tmp

COPY packages/requirements.txt packages/cisl-gpu-base.yml /tmp/

RUN ${CONDA_DIR}/bin/pip install --no-cache -r requirements.txt

COPY --chown="${NB_UID}:${NB_GID}" configs/.condarc "${CONDA_DIR}/.condarc"

RUN CONDA_OVERRIDE_CUDA="12.3" mamba env create --name ${CONDA_ENV} -f cisl-gpu-base.yml \
    && mamba clean -afy \
    # Fix permissions
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "${CONDA_USR_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# Run conda activate each time a bash shell starts, so users don't have to manually type conda activate
# Note this is only read by shell, but not by the jupyter notebook - that relies
# on us starting the correct `python` process, which we do by adding the notebook conda environment's
# bin to PATH earlier ($NB_PYTHON_PREFIX/bin)
# GPU TensorFlow config based on https://github.com/tensorflow/tensorflow/issues/58681
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh ; conda activate ${CONDA_ENV}" > /etc/profile.d/init_conda.sh&& \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_DIR}/lib/' > ${CONDA_DIR}/etc/conda/activate.d/env_vars.sh && \
    printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CONDA_DIR}/lib/\n' >> ${CONDA_DIR}/etc/conda/activate.d/env_vars.sh

COPY configs/jupyter_server_config.py /etc/jupyter/jupyter_server_config.py
# Used to allow user deletions of folders and contents
RUN sed -i 's/c.FileContentsManager.delete_to_trash = False/c.FileContentsManager.always_delete_dir = True/g' /etc/jupyter/jupyter_server_config.py

# Make the conda environments we install read only and executable for the user
# They can run the environments but will get permission denied when trying to make changes
# New environments are installed to /home/jovyan/.jupyter with write permissions for the users
RUN chmod 755 /srv/base-conda/cisl-gpu-base/* && \
    chown root:root /srv/*   

USER ${NB_USER}
WORKDIR ${HOME}

COPY --chmod=755 /scripts/start /srv/start

EXPOSE 8888
ENTRYPOINT ["/srv/start"]
