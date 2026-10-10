import logging
import hashlib
import json
import os
import time
from ruamel.yaml import YAML
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

from transformers import AutoModelForSequenceClassification


# import PyPDF2
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
import files
import fitz  # PyMuPDF


class TextPreparer:
    def __init__(self):
        self.input_file = ""
        self.output_file = ""

    def set(self, input_file, output_file):
        if not input_file:
            raise ValueError("Input PDF path is required.")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"PDF file not found: {input_file}")
        if not output_file:
            raise ValueError("Output TXT path is required.")
        self.input_file = input_file
        self.output_file = output_file
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        logging.info(f"pdf: {self.input_file} text: {self.output_file}")

    def pdf_to_text(self):
        if not self.input_file or not os.path.exists(self.input_file):
            raise FileNotFoundError(f"PDF file not found: {self.input_file}")
        logging.info(f"Converting pdf to text: {self.input_file}")
        doc = fitz.open(self.input_file)
        full_text = []
        for page_num, page in enumerate(doc):
            text_blocks = page.get_text("blocks")
            page_height = page.rect.height
            page_text = []
            for block in text_blocks:
                x0, y0, x1, y1, text, *_ = block

                if y0 < page_height * 0.08 or y1 > page_height * 0.90:
                    continue

                if text.strip().isdigit() and len(text.strip()) < 5:
                    continue

                page_text.append(text.strip())

            full_text.append("\n".join(page_text))

        # Write to TXT file
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
        logging.info(f"Saved to: {self.output_file}")

    def pdf_to_chunks(self, chunk_size=1000, chunk_overlap=200):
        """Convert PDF to text chunks for RAG."""
        self.pdf_to_text()
        with open(self.output_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_text(raw_text)
        return chunks


class KnowledgeBase:
    def __init__(
        self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", revision=None
    ):
        self.model_name = model_name
        self.model_revision = revision
        self.embedder = SentenceTransformer(model_name, revision=revision)
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        if not chunks:
            raise ValueError("Cannot build FAISS index from an empty chunk list.")
        self.chunks = chunks
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        logging.info(f"FAISS index built with {len(chunks)} chunks")

    def save(self, path="kodeks_index.faiss", chunks_path="chunks.pkl"):
        if self.index is None:
            raise ValueError("Cannot save FAISS index because it has not been built.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        self.index_path = path

    def load(self, path="kodeks_index.faiss", chunks_path="chunks.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index file not found: {path}")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"FAISS chunk file not found: {chunks_path}")
        self.index = faiss.read_index(path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.index_path = path

    def load_if_exists(self, path="kodeks_index.faiss", chunks_path="chunks.pkl"):
        if os.path.exists(path) and os.path.exists(chunks_path):
            self.load(path=path, chunks_path=chunks_path)
            logging.info(
                f"Loaded existing FAISS index from {path} and chunks from {chunks_path}"
            )
            return True
        return False

    def retrieve(self, query, top_k=3):
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        retrieved = []
        for rank, (distance, chunk_index) in enumerate(
            zip(distances[0], indices[0]), start=1
        ):
            if chunk_index < 0:
                continue
            chunk_index = int(chunk_index)
            chunk_text = self.chunks[chunk_index]
            retrieved.append(
                {
                    "rank": rank,
                    "chunk_index": chunk_index,
                    "distance": float(distance),
                    "chunk_sha256": hashlib.sha256(
                        chunk_text.encode("utf-8")
                    ).hexdigest(),
                    "chunk_text": chunk_text,
                }
            )
        return retrieved


class TextClassifier:
    def __init__(self, model_id, knowledge_base=None):
        self._setup_logging(model_id)
        self._setup_yaml()
        self._read_configuration_file()

        if model_id not in self._configuration_map["llm"]["models"]:
            raise KeyError(f"MODEL_ID '{model_id}' is not defined in environment.yml")

        # Set system attributes
        self.hf_token = self._configuration_map["llm"]["system"]["huggingface_token"]

        # Set common attributes
        self.model_revision = self._configuration_map["llm"]["common"]["model_revision"]
        self.model_dtype = getattr(
            torch, self._configuration_map["llm"]["common"]["model_dtype"].removeprefix("torch.")
        )
        self.torch_dtype = getattr(
            torch, self._configuration_map["llm"]["common"]["torch_dtype"].removeprefix("torch.")
        )
        generation = self._configuration_map["llm"]["common"]["generation"]
        self.generation_seed = generation["seed"]
        self.generation_overrides = {
            key: generation[key]
            for key in ("max_new_tokens", "pad_token_id", "eos_token_id", "temperature", "top_p", "top_k", "do_sample")
        }
        torch.manual_seed(self.generation_seed)

        # Set specific model attributes
        model_data = self._configuration_map["llm"]["models"][model_id]

        self.model_id = model_id
        self.model_name = model_data["model_name"]
        self.model_name_basic = model_data["model_name_basic"]
        self.input_csv = model_data["input_csv"]
        self.chunks_no = model_data["chunks_no"]
        self.prompt_id = model_data["prompt_id"]
        self.use_rag = model_data.get("use_rag", True)

        if self.use_rag:
            self.knowledge_base = knowledge_base
        else:
            self.logger.info(f"Knowledge base set to None since use_rag is False")
            self.knowledge_base = None

        self.output_csv = f"/output/{model_id}.csv"
        self.retrieval_log_path = f"/output/{model_id}_retrievals.jsonl"
        self.device = torch.device("cpu")
        self.logger.info(f"Device: {self.device}")
        if self.hf_token:
            self._login_huggingface()

        self.logger.info("Init tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.model_revision
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id != self.generation_overrides["pad_token_id"]:
            raise ValueError("Configured pad_token_id does not match the tokenizer.")
        self.logger.info("Init model (CausalLM)")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=self.model_revision, torch_dtype=self.torch_dtype
        ).to(device=self.device, dtype=self.model_dtype)
        self._validate_runtime_configuration()
        self.logger.info("Init completed")

    def _validate_runtime_configuration(self):
        model_revision = getattr(self.model.config, "_commit_hash", None)
        tokenizer_revision = getattr(self.tokenizer, "init_kwargs", {}).get(
            "_commit_hash"
        )
        if model_revision is None:
            model_revision = tokenizer_revision
        effective_generation_config = self.model.generation_config.to_dict()
        effective_generation_config.update(self.generation_overrides)
        common = self._configuration_map["llm"]["common"]
        profile = {
            "model_id": self.model_id,
            "model_id_env": os.getenv("MODEL_ID", "not_set"),
            "model_name": self.model_name,
            "model_name_basic": self.model_name_basic,
            "model_revision": model_revision or "unknown",
            "configured_model_revision": self.model_revision,
            "tokenizer_revision": tokenizer_revision or "unknown",
            "input_csv": self.input_csv,
            "output_csv": self.output_csv,
            "device": self.device,
            "use_rag": self.use_rag,
            "chunks_no": self.chunks_no,
            "prompt_id": self.prompt_id,
            "knowledge_base_enabled": self.knowledge_base is not None,
            "embedding_model": common["embedding_model"],
            "embedding_revision": common["embedding_revision"],
            "chunk_size": common["chunk_size"],
            "chunk_overlap": common["chunk_overlap"],
            "model_dtype": getattr(self.model, "dtype", "unknown"),
            "configured_model_dtype": self.model_dtype,
            "torch_dtype": getattr(self.model.config, "torch_dtype", None),
            "configured_torch_dtype": self.torch_dtype,
            "quantization_config": getattr(self.model.config, "quantization_config", None),
            "load_in_4bit": getattr(self.model.config, "load_in_4bit", None),
            "load_in_8bit": getattr(self.model.config, "load_in_8bit", None),
            "model_type": getattr(self.model.config, "model_type", "unknown"),
            "hidden_size": getattr(self.model.config, "hidden_size", "unknown"),
            "vocab_size": getattr(self.model.config, "vocab_size", "unknown"),
            "num_hidden_layers": getattr(self.model.config, "num_hidden_layers", "unknown"),
            "num_attention_heads": getattr(self.model.config, "num_attention_heads", "unknown"),
            "max_position_embeddings": getattr(self.model.config, "max_position_embeddings", "unknown"),
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "tokenizer_vocab_size": getattr(self.tokenizer, "vocab_size", "unknown"),
            "tokenizer_chat_template_available": bool(getattr(self.tokenizer, "chat_template", None)),
            "chat_template_applied": False,
            "effective_generation_config": effective_generation_config,
            "application_random_seed": self.generation_seed,
            "python_hash_seed": os.getenv("PYTHONHASHSEED", "not_set"),
            "torch_initial_seed": torch.initial_seed(),
        }
        self.logger.info("Effective execution profile:")
        for key, value in profile.items():
            self.logger.info("  %s: %s", key, value)

        if self.model is None:
            raise RuntimeError("Model is None after loading.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is None after loading.")
        if self.model_name is None or self.model_name == "":
            raise ValueError("Model name is empty after loading.")
        if self.use_rag and (self.knowledge_base is None or self.chunks_no <= 0):
            raise ValueError(
                "RAG is enabled but knowledge base or chunks_no is invalid."
            )
        if self.use_rag and self.knowledge_base is not None and self.knowledge_base.index is None:
            raise ValueError("RAG is enabled but the FAISS index is not initialized.")

    def _setup_logging(self, model_id):
        log_path = f"/logs/{model_id}.log"
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create a StreamHandler (console output)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # Create a FileHandler (file output)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _setup_yaml(self):
        self._yaml = YAML()
        self._yaml.indent(mapping=4, sequence=6, offset=3)
        self._yaml.explicit_start = True
        self._yaml.width = 16384

    def _read_configuration_file(self):
        with open("environment.yml", "r") as text_file:
            self._configuration_map = self._yaml.load(text_file)
        self.logger.info("configuration.yaml loaded")
        self.logger.info(self._configuration_map)

    def _login_huggingface(self):
        try:
            login(token=self.hf_token)
            self.logger.info("Logged in to Hugging Face.")
        except Exception as e:
            self.logger.error(f"Failed to login to Hugging Face: {e}")

    def _write_retrieval_record(self, case_id, query_text, retrieved, retrieval_performed):
        record = {
            "record_type": "retrieval",
            "model_id": self.model_id,
            "input_id": str(case_id),
            "query_text": query_text,
            "use_rag": self.use_rag,
            "top_k_requested": self.chunks_no if self.use_rag else 0,
            "retrieval_performed": retrieval_performed,
            "embedding_model": (
                self.knowledge_base.model_name
                if self.knowledge_base is not None
                else None
            ),
            "distance_metric": "squared_l2",
            "retrieved_chunks": retrieved,
        }
        with open(self.retrieval_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_prompt(self, text, context):
        if self.prompt_id == 1:
            instruction = "Odpowiedz, czy czyn opisany w tekście jest zabroniony przez polski Kodeks Karny, czy nie. Tylko odpowiedź TAK lub NIE. Bez uzasadnienia."
        elif self.prompt_id == 2:
            instruction = "Odpowiedz, czy czyn opisany w tekście jest zabroniony przez polski Kodeks Karny, czy nie. Jeśli tak, rozpocznij odpowiedź od TAK, a następnie podaj właściwy przepis Kodeksu Karnego; jeśli nie, odpowiedz tylko NIE. Bez uzasadnienia."
        elif self.prompt_id == 3:
            instruction = "Przyjmij następujące znaczenie etykiet: TAK oznacza, że opisany czyn jest zabroniony przez polski Kodeks Karny; NIE oznacza, że opisany czyn nie jest zabroniony przez polski Kodeks Karny.\nOdpowiedz, czy czyn opisany w tekście jest zabroniony przez polski Kodeks Karny, czy nie. Tylko odpowiedź TAK lub NIE. Bez uzasadnienia."
        else:
            raise ValueError(f"Unsupported prompt_id: {self.prompt_id}")

        context_prefix = f"Kontekst (fragmenty Kodeksu Karnego):\n{context}\n" if context else ""
        return (
            f"{context_prefix}"
            f"Zaklasyfikuj poniższe wydarzenie na podstawie polskiego Kodeksu Karnego: '{text}'.\n"
            f"{instruction}"
        )

    def _log_inference_timing(self, case_id, step, started_at):
        self.logger.info(
            "Inference timing: input_id=%s step=%s elapsed_seconds=%.6f",
            case_id,
            step,
            time.perf_counter() - started_at,
        )

    def classify_text(self, text, case_id=None):
        inference_started_at = time.perf_counter()
        try:
            self.logger.info(
                f"Classify text: {text} -> with depth: {self.chunks_no} and knowledge base: {self.knowledge_base is not None}"
            )

            step_started_at = time.perf_counter()
            context = ""
            retrieval_performed = (
                self.knowledge_base is not None
                and self.knowledge_base.index is not None
                and self.use_rag
            )
            retrieved = []
            if retrieval_performed:
                retrieved = self.knowledge_base.retrieve(text, top_k=self.chunks_no)
                context = "\n".join(
                    chunk["chunk_text"] for chunk in retrieved
                )
            self._log_inference_timing(case_id, "retrieval", step_started_at)

            step_started_at = time.perf_counter()
            self._write_retrieval_record(case_id, text, retrieved, retrieval_performed)
            self._log_inference_timing(case_id, "retrieval_record_write", step_started_at)

            step_started_at = time.perf_counter()
            prompt = self._build_prompt(text, context)
            self._log_inference_timing(case_id, "prompt_construction", step_started_at)

            step_started_at = time.perf_counter()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_token_count = inputs["input_ids"].shape[-1]
            self._log_inference_timing(case_id, "tokenization", step_started_at)
            model_context_limit = getattr(
                self.model.config, "max_position_embeddings", "unknown"
            )
            self.logger.info(
                f"Prompt length: characters={len(prompt)}, "
                f"input_tokens={input_token_count}, "
                f"model_context_limit={model_context_limit}, truncation=False"
            )
            step_started_at = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **self.generation_overrides,
                )
            self._log_inference_timing(case_id, "generation", step_started_at)

            step_started_at = time.perf_counter()
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if result.startswith(prompt):
                result = result[len(prompt) :].strip()
            result = result.replace("\n", " | ")
            self._log_inference_timing(case_id, "decoding", step_started_at)
            self.logger.debug(f"Model's Result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error occurred during classification: {e}")
            return "Error"
        finally:
            self._log_inference_timing(case_id, "total", inference_started_at)

    def process_csv_and_save_csv(self):
        try:
            if not os.path.exists(self.input_csv):
                raise FileNotFoundError(f"Input CSV file not found: {self.input_csv}")
            df = pd.read_csv(self.input_csv, sep=";")
            required_columns = {"ID", "Notatka"}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise KeyError(
                    f"Input CSV is missing required columns: {sorted(missing_columns)}"
                )
            self.logger.info("Input data csv loaded")
            source_documents = []
            for source_id, source_paths in files.files_dir.items():
                with open(source_paths["pdf_file"], "rb") as source_file:
                    source_documents.append(
                        {
                            "source_id": source_id,
                            "pdf_sha256": hashlib.file_digest(source_file, "sha256").hexdigest(),
                        }
                    )
            if self.knowledge_base is not None:
                with open(self.knowledge_base.index_path, "rb") as index_file:
                    index_sha256 = hashlib.file_digest(index_file, "sha256").hexdigest()
                chunk_hashes = [
                    hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    for chunk in self.knowledge_base.chunks
                ]
            else:
                index_sha256 = None
                chunk_hashes = []
            with open(self.retrieval_log_path, "w", encoding="utf-8") as log_file:
                log_file.write(
                    json.dumps(
                        {
                            "record_type": "run",
                            "model_id": self.model_id,
                            "model_name": self.model_name,
                            "model_revision": getattr(
                                self.model.config, "_commit_hash", None
                            ),
                            "embedding_model": (
                                self.knowledge_base.model_name
                                if self.knowledge_base is not None
                                else None
                            ),
                            "embedding_revision": (
                                self.knowledge_base.model_revision
                                if self.knowledge_base is not None
                                else None
                            ),
                            "source_corpus": "KodeksKarny",
                            "source_documents": source_documents,
                            "chunk_size": self._configuration_map["llm"]["common"]["chunk_size"],
                            "chunk_overlap": self._configuration_map["llm"]["common"]["chunk_overlap"],
                            "index_sha256": index_sha256,
                            "chunk_count": len(chunk_hashes),
                            "chunk_sha256": chunk_hashes,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            results = []
            for _, row in df.iterrows():
                text = row["Notatka"]
                text_id = row["ID"]
                result = self.classify_text(text, case_id=text_id)
                results.append(
                    {
                        "ID": text_id,
                        "Notatka": text,
                        f"{self.model_name_basic}(treść)": result,
                    }
                )
                result_df = pd.DataFrame(results)
                result_df.to_csv(self.output_csv, index=False, sep=";")
            self.logger.info(f"Results saved to {self.output_csv}")
        except FileNotFoundError:
            self.logger.error(f"Input CSV file not found: {self.input_csv}")
        except Exception as e:
            self.logger.error(f"An error occurred during CSV processing: {e}")


def main():
    logging.basicConfig(level=logging.DEBUG)

    if not files.files_dir:
        raise FileNotFoundError(
            "No corpus files configured. Expected PDF/TXT pairs under original_documents and text_documents."
        )

    text_prep = TextPreparer()
    with open(os.path.join(os.path.dirname(__file__), "environment.yml"), "r") as config_file:
        common = YAML().load(config_file)["llm"]["common"]
    kb = KnowledgeBase(
        model_name=common["embedding_model"], revision=common["embedding_revision"]
    )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_id = f"{common['embedding_revision'][:12]}_{common['chunk_size']}_{common['chunk_overlap']}"
    index_path = os.path.join(project_root, f"kodeks_index_{index_id}.faiss")
    chunks_path = os.path.join(project_root, f"chunks_{index_id}.pkl")

    if not kb.load_if_exists(path=index_path, chunks_path=chunks_path):
        logging.info("FAISS index not found, rebuilding knowledge base...")
        all_chunks = []
        for i in files.files_dir:
            pdf_path = files.files_dir[i]["pdf_file"]
            text_path = files.files_dir[i]["txt_file"]
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Required PDF file not found: {pdf_path}")
            if not os.path.exists(text_path):
                raise FileNotFoundError(f"Required TXT file not found: {text_path}")
            text_prep.set(pdf_path, text_path)
            chunks = text_prep.pdf_to_chunks(
                chunk_size=common["chunk_size"], chunk_overlap=common["chunk_overlap"]
            )
            if not chunks:
                raise ValueError(f"No chunks generated from PDF: {pdf_path}")
            all_chunks.extend(chunks)
        kb.build_index(all_chunks)
        kb.save(path=index_path, chunks_path=chunks_path)

    # prepare model
    model_id = os.getenv("MODEL_ID", "unknown")
    print(f"main: MODEL_ID: {model_id}")

    if model_id == "unknown":
        raise ValueError("MODEL_ID environment variable not set.")
    try:
        knowledge_base_instance = kb
        classifier = TextClassifier(model_id, knowledge_base=kb)
        classifier.process_csv_and_save_csv()
    except Exception as e:
        print(f"An error occurred during text classification: {e}")


if __name__ == "__main__":
    main()
