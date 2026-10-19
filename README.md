# Evaluating RAG-Based Legal Text Classification in Polish with Bielik-11B

This repository contains code and configuration to run experiments to assess the use of RAG augmentation for classifying legal texts.

## Tests architecture

To provide a static configuration, tests are executed using Docker containers. Containers are built with an enumerated list of libraries to achieve repeatability.

## Repository structure

The repository contains:

- execution scripts `models_*.sh`
- `code` directory containing
  - application in `code/application` directory
  - entry dataset as csv file in `code/data` directory
  - RAG sources as pdf file in `code/original_documents` directory
    - `KodeksKarny.pdf` file with Polish Penal Code
- `inferences` directory with tests results
  - `smoke_set` contains full set of data, logs and calculations for four elements simplified input
  - `full_set` contains subset of data (not containing dataset details) of full passage of inferences and statistical analysis

## Test configuration

All test configuration is placed in `code/app/environment.yml` file. It contains:

- Access token for the Hugging Face portal to download models

```yaml
llm:
  system:
    huggingface_token: redacted
```

Working `hugginface_token` is required to download Bielik model.

- Common model settings

```yaml
llm:
  common:
    model_revision: 73f5f7fb9a79a4409d02740155a3868c1ff0d2bc
    model_dtype: torch.float16
    torch_dtype: torch.float16
    embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    embedding_revision: e8f8c211226b894fcb81acc59f3b34ba3efd5f42
    chunk_size: 1000
    chunk_overlap: 200
    generation:
      max_new_tokens: 250
      pad_token_id: 2
      eos_token_id: [32001, 2]
      temperature: 1.0
      top_p: 1.0
      top_k: 50
      do_sample: false
      seed: 10622516036899522780
```

- Test scenario definitions (set of scenarios with syntax like this)

```yaml
llm:
  models:
    bielik11-268-cpu-rag-20-prompt-1:
      model_name_basic: Bielik
      model_name: speakleash/Bielik-11B-v2.3-Instruct
      input_csv: /data/dataset_268.csv
      chunks_no: 20
      use_rag: true
      prompt_id: 1
```

The scenario identifier is used to name output files. It contains:

- prefix `bielik11-4-cpu` - has contain set size (here 4)
- scenario i.e. `rag-20`, `norag`
- prompt id i.e. `prompt-1`. Prompt is embedded in main.py file and is chosen by id.

## Text execution

The following scripts can be used to maintain the tests. Their contents can be modified to match the planned test executions.

### Models stop

The `models_stop.sh` script can stop and destroy all currently running tests.

The script contains commands to stop and remove containers. `MODEL_ID` is an identifier defined in the configuration.

```bash
MODEL_ID=bielik11-268-cpu-norag-prompt-1 ./code/app/stop.sh
```

### Models build

The `models_build.sh` script can rebuild the Docker images used by the tests.

Two kinds of images are created:

- pytorch image `pytorch/pytorch`
- test images i.e. `llm-classifier-bielik11-268-cpu-rag-10-prompt-1`.

Each test configuration has its own image.

The script contains commands to build containers.

```bash
MODEL_ID=bielik11-268-cpu-norag-prompt-1 ./code/app/build.sh
```

### Models run

Script `models_run.sh` provides ability to run tests.

```bash
MODEL_ID=bielik11-268-cpu-norag-prompt-1 ./code/app/start.sh
```

## Test execution output

Tests create the following artifacts:

- inference output in `code/output` directory as csv file
- inference auxiliary information in `code/output` as a JSONL file
- log file from inference in `code/logs` directory

Each scenario produces its own set of files, named with the scenario ID prefix.

## Run example

To run the example inferences, run the script (parameters have to be adjusted to current set):

```bash
./validation.sh
```

This script uses a smoke-test data file with 4 cases (2 positive and 2 negative, specially created for this test).

- the data file is copied into the `code/data` directory
- inference images are built
- inference is run
- output artifacts are copied into the `example/01.raw_data` directory.

## Statistical analysis

Statistical analysis can be created by running script `run_postprocessing.sh`. Set size has to be set in `parameters.py` file.

The `example/02_processed_data` directory contains raw data with posprocessed data usable for inference assessment.

All data is stored in CSV files.

| Directory | Content |
| --- | --- |
| `01.key` | Classification key file |
| `02.inference_output` | Raw inference output |
| `03.cleared_inference_output` | Normalized inference output with filtered original query |
| `04.classification` | Classification results (TP, TN, FP, FN, ERROR, MISSING) |
| `05.classification_errors` | Error cases containing no result, empty, ... |
| `06.summarized results` | Results for each scenario (no-RAG, k=3, k=5, k=7, k=10 and k=20) combined for calculations |
| `10.inference_logs` | Raw inference logs |
| `11.inference_stats` | Inferences time execution metrics |
| `12.retrieval_validation` | Statistics for retrieval validation |
| `13.output_quality` | Statistics of well formed outputs and outputs which require post-processing |
| `14.statistics` | Statistics including calculated MacNemar, Holm and other calculations |
| `15.ablation_analysis` | Ablation analysis results |
