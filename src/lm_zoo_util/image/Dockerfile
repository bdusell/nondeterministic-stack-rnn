FROM renormalizing-ns-rnn
COPY . /app/model
ARG VERSION
ARG NAME
RUN checksum=$(tar c src model | sha1sum | cut -d ' ' -f 1) && \
    cd src && \
    poetry run python lm_zoo_util/write_spec.py \
        --version "$VERSION" \
        --checksum "$checksum" \
        --name "$NAME" \
        --url http://example.com/ \
        --vocab "$(cat /app/model/vocab-file.txt)" \
        --dataset "$(cat /app/model/dataset.txt)" \
        > /app/src/lm_zoo_util/bin/spec.json
ENV PATH /app/src/lm_zoo_util/bin:${PATH}
