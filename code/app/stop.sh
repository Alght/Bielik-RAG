#!/bin/bash -e

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

cd ${SCRIPT_DIR}

echo "Stop docker container llm-classifier-${MODEL_ID}"
set +e
docker stop llm-classifier-${MODEL_ID}
set -e

echo "Remove docker container llm-classifier-${MODEL_ID}"
set +e
docker rm llm-classifier-${MODEL_ID} -f
set -e

cd ${CURRENT_DIR}
