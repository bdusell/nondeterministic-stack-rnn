FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
# UTF-8 encoding is necessary for printing non-ASCII characters to the
# terminal.
ENV LC_ALL C.UTF-8
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3.8-venv \
        curl \
        texlive-xetex && \
    rm -rf /var/lib/apt/lists/*
# Symlink `python` to `python3.8`.
# See https://github.com/sdispater/poetry/issues/721
RUN ln -s "$(which python3.8)" /usr/local/bin/python
# Install Poetry.
RUN cd /tmp && \
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
        > get-poetry.py && \
    POETRY_HOME=/usr/local/poetry python get-poetry.py -y && \
    rm get-poetry.py && \
    chmod 755 /usr/local/poetry/bin/poetry
ENV PATH /usr/local/poetry/bin:${PATH}
# Stores Python packages in the local directory.
# See https://python-poetry.org/docs/configuration/#virtualenvsin-project-boolean
ENV POETRY_VIRTUALENVS_IN_PROJECT true
ENV PYTHONPATH /app/src:${PYTHONPATH}
WORKDIR /app/
