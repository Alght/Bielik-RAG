#! /bin/bash

echo "--------------------------------------"
echo "Stop all containers and clear old runs"
echo "--------------------------------------"

mkdir -p ./code/output
mkdir -p ./code/logs
rm -f ./code/output/*
rm -f ./code/logs/*
./models_stop.sh

echo "--------------------------------------"
echo "Build all containers"
echo "--------------------------------------"

./models_build.sh

echo "--------------------------------------"
echo "Copy initial dataset"
echo "--------------------------------------"

mkdir -p ./code/app/data

cp ./example/01.raw_data/dataset_4.csv ./code/app/data

echo "--------------------------------------"
echo "Run all inferences sequentially"
echo "--------------------------------------"

./models_run.sh

echo "--------------------------------------"
echo "Remove all containers (if necessary)"
echo "--------------------------------------"

./models_stop.sh

echo "--------------------------------------"
echo "Copy inference artifacts"
echo "--------------------------------------"

mkdir -p ./example/01.raw_data/02.inference_output
mkdir -p ./example/01.raw_data/10.inference_logs

cp ./code/output/* ./example/01.raw_data/02.inference_output
cp ./code/logs/* ./example/01.raw_data/10.inference_logs

