#!/bin/bash -e

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

if [[ "${MODEL_ID}" == "" ]]; then
    echo "Environment variable MODEL_ID is not set"
    exit 1
fi

cd ${SCRIPT_DIR}

source ./proxy.sh

echo "Build docker with model ${MODEL_ID}"

docker build \
        --build-arg HTTP_PROXY=${HTTP_PROXY} \
        --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
        --build-arg HTTP_PROXY=${HTTPS_PROXY} \
        --build-arg NO_PROXY=${NO_PROXY} \
        --build-arg http_proxy=${http_proxy} \
        --build-arg https_proxy=${https_proxy} \
        --build-arg no_proxy=${no_proxy} \
        -t llm-classifier-${MODEL_ID}:latest .
