# Evaluating RAG-Based Legal Text Classification in Polish with Bielik-11B

This repository contains code and configuration to run experiments to asses usage of RAG augmentation for classification of legal texts.

## Tests architecture

To provide static configuration tests are executed using docker containers. Containers are build with enumerated list of libraries to achieve repetability.

## Repository structure

Repository contains:

- execution scripts `models_*.sh`
- `code` directory containinng
  - application in `code/application` directory
  - entry dataset as csv file in `code/data` directory
  - RAG sources as pdf file in `code/original_documents` directory
    - `KodeksKarny.pdf` file with Polish Penal Code

## Test configuration

All test configuration is placed in `code/app/environment.yml` file. It contains:

- Access token to HuggingFace portal to download models

```yaml
llm:
  system:
    huggingface_token:
```

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

- Test scenarios definitions

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

Identifier of the scenario is used to name output files. It contains:

- prefix `bielik11-268-cpu` - can be set freely
- scenario i.e. `rag-20`, `norag`
- prompt id i.e. `prompt-1`. Prompt is embedded in main.py file and is chosen by id.

## Text execution

To maintain tests the following scripts can be used. The content of the scripts can be modified accordingly to current test executions planned.

### Models stop

Script `models_stop.sh` provides ability to stop and destroy all currently run tests.

Script contains lines which stop the containers and remove them. MODEL_ID is the identifier existing in configuration.

```bash
MODEL_ID=bielik11-268-cpu-norag-prompt-1 ./code/app/stop.sh
```

### Models build

Script `models_build.sh` privides ability to rebuild docker images used by tests.

There are two kind of images created:

- pytorch image `pytorch/pytorch`
- test images i.e. `llm-classifier-bielik11-268-cpu-rag-10-prompt-1`.

Each test configuration has own image. In general there are only tags to one common image.

Script contains lines which build containers.

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
- inverence auxialary information in `code/output` as jsonl file
- log file from inference in `code/logs` directory

Each scenario produces own file set named with scenario id prefix.
