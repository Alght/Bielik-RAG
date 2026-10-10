#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

cd "${SCRIPT_DIR}"

retries=0
max_retries=50

export DOCKER_BUILDKIT=0

while true; do
    ./start.sh "${MODEL_ID}"
    if [ $? -eq 0 ]; then
        echo "start.sh ${MODEL_ID} executed successfully."
        break
    else
        echo "start.sh ${MODEL_ID} failed with exit code $?. Retrying..."
        ./stop.sh "${MODEL_ID}"
        sleep 5
        retries=$((retries + 1))
        if [ "$retries" -ge "$max_retries" ]; then
            echo "Maximum retry count ($max_retries) reached. Exiting."
            exit 1 # Exit with a non-zero code to indicate failure
        fi
    fi
done

echo "Script finished successfully."
