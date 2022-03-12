FROM renormalizing-ns-rnn-dev:latest
WORKDIR /app/
COPY pyproject.toml poetry.lock .
RUN poetry install
COPY data/mikolov-ptb/vocab.txt ./data/mikolov-ptb/vocab.txt
COPY src ./src
