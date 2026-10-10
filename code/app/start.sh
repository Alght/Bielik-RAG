#!/bin/bash -e

if [[ "${MODEL_ID}" == "" ]]; then
    echo "Environment variable MODEL_ID is not set"
    exit 1
fi

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

cd ${SCRIPT_DIR}

source ./proxy.sh

echo "Run docker with model ${MODEL_ID}"

docker run \
    --name llm-classifier-${MODEL_ID} \
    --env MODEL_ID=${MODEL_ID} \
    --env HTTP_PROXY=${HTTP_PROXY} \
    --env HTTPS_PROXY=${HTTPS_PROXY} \
    --env NO_PROXY=${NO_PROXY} \
    --env http_proxy=${HTTP_PROXY} \
    --env https_proxy=${HTTPS_PROXY} \
    --env no_proxy=${NO_PROXY} \
    --volume "$(pwd)/../data:/data" \
    --volume "$(pwd)/../cache:/root/.cache/huggingface" \
    --volume "$(pwd)/../logs:/logs" \
    --volume "$(pwd)/../app:/app" \
    --volume "$(pwd)/../original_documents:/original_documents" \
    --volume "$(pwd)/../text_documents:/text_documents" \
    --volume "$(pwd)/../output:/output" \
    llm-classifier-${MODEL_ID}:latest \
    python main.py
