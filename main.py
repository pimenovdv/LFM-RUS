import os
import click
import subprocess
import yaml
import sys
from lfm_rus.tokenizer import train_tokenizer
from lfm_rus.lexical_init import initialize_lexical_embeddings
from lfm_rus.pruning import prune_tokenizer_and_model
from lfm_rus.embedding_warmup import embedding_warmup
from datasets import load_dataset

def create_axolotl_config(
    model_name_or_path: str,
    dataset: str,
    output_dir: str,
    context_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    is_sft: bool = False
) -> dict:
    dataset_config = {
        "path": dataset,
        "type": "completion" if not is_sft else "alpaca"
    }

    config = {
        "base_model": model_name_or_path,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "load_in_8bit": False,
        "load_in_4bit": False,
        "strict": False,
        "datasets": [dataset_config],
        "dataset_prepared_path": os.path.join(output_dir, "data_prepared"),
        "val_set_size": 0.05,
        "output_dir": output_dir,
        "sequence_len": context_length,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "wandb_project": "lfm-rus" if os.environ.get("WANDB_API_KEY") else "",
        "gradient_accumulation_steps": 4,
        "micro_batch_size": batch_size,
        "num_epochs": epochs,
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": learning_rate,
        "train_on_inputs": False,
        "group_by_length": False,
        "bf16": True,
        "fp16": False,
        "tf32": False,
        "gradient_checkpointing": True,
        "early_stopping_patience": None,
        "resume_from_checkpoint": None,
        "local_rank": None,
        "logging_steps": 1,
        "xformers_attention": False,
        "flash_attention": True,
        "warmup_steps": 10,
        "evals_per_epoch": 4,
        "saves_per_epoch": 1,
        "debug": False,
        "deepspeed": None,
        "weight_decay": 0.0,
        "fsdp": [],
        "fsdp_config": None,
        "special_tokens": {
            "pad_token": "<|endoftext|>"
        }
    }
    return config

def run_axolotl(config_path: str):
    print(f"Running Axolotl with config: {config_path}")
    cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Axolotl training failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find 'accelerate' or 'axolotl'. Ensure they are installed.")
        sys.exit(1)

def get_warmup_texts(dataset_name, max_samples=10000):
    try:
        dataset = load_dataset(dataset_name, split="train")
        texts = [item["text"] for item in dataset.select(range(min(len(dataset), max_samples)))]
        return texts
    except Exception as e:
        print(f"Error loading dataset {dataset_name} for warmup: {e}")
        return []

def run_pruning(pipeline_config: dict, stage_config: dict):
    print("\n--- Stage 0: Pruning ---")
    current_model = pipeline_config.get("model", "gpt2")
    output_dir = pipeline_config.get("output_dir", "./output")

    tokens_to_prune = stage_config.get("tokens_to_prune", "")
    prune_list = [t.strip() for t in tokens_to_prune.split(",") if t.strip()]

    if prune_list:
        print(f"Pruning tokens: {prune_list}")
        prune_output_dir = os.path.join(output_dir, "model_pruned")
        prune_tokenizer_and_model(
            model_name=current_model,
            tokens_to_remove=prune_list,
            save_path=prune_output_dir
        )
        return prune_output_dir
    else:
        print("No tokens to prune provided. Skipping pruning.")
        return current_model

def run_tokenizer(pipeline_config: dict, stage_config: dict):
    print("\n--- Stage 1: Tokenizer Training & Lexical Initialization ---")
    current_model = pipeline_config.get("model", "gpt2")
    output_dir = pipeline_config.get("output_dir", "./output")

    dataset_tokenizer = stage_config.get("dataset_tokenizer", "IlyaGusev/ru_instruct")
    new_tokens = stage_config.get("new_tokens", 10000)

    print(f"Training tokenizer on dataset: {dataset_tokenizer}")
    added_tokens, tokenizer = train_tokenizer(
        base_model_name=current_model,
        dataset_name=dataset_tokenizer,
        new_tokens=new_tokens
    )
    print(f"Added {len(added_tokens)} new tokens.")

    tokenizer_output_dir = os.path.join(output_dir, "model_with_tokenizer")
    print(f"Initializing lexical embeddings and saving to: {tokenizer_output_dir}")
    _, _ = initialize_lexical_embeddings(
        model_name=current_model,
        new_tokens=added_tokens,
        save_path=tokenizer_output_dir
    )

    # Store added tokens in pipeline_config so warmup can use it
    if "added_tokens" not in pipeline_config:
        pipeline_config["added_tokens"] = []
    pipeline_config["added_tokens"].extend(added_tokens)

    return tokenizer_output_dir

def run_warmup(pipeline_config: dict, stage_config: dict):
    print("\n--- Stage 1.5: Embedding Warm-up ---")
    current_model = pipeline_config.get("model", "gpt2")
    output_dir = pipeline_config.get("output_dir", "./output")
    batch_size = pipeline_config.get("batch_size", 4)
    learning_rate = pipeline_config.get("learning_rate", 2e-5)

    dataset_warmup = stage_config.get("dataset_warmup", "IlyaGusev/ru_instruct")
    epochs_warmup = stage_config.get("epochs_warmup", 1)

    added_tokens = pipeline_config.get("added_tokens", [])

    warmup_output_dir = os.path.join(output_dir, "model_warmup")
    print(f"Fetching texts for warmup from {dataset_warmup}...")
    texts = get_warmup_texts(dataset_warmup)

    if texts:
        embedding_warmup(
            model_name_or_path=current_model,
            texts=texts,
            new_tokens=added_tokens,
            epochs=epochs_warmup,
            batch_size=batch_size,
            lr=learning_rate,
            save_path=warmup_output_dir
        )
        return warmup_output_dir
    else:
        print("No texts found for warmup. Skipping Embedding Warm-up.")
        return current_model

def run_axolotl_stage(pipeline_config: dict, stage_config: dict, stage_name: str, is_sft: bool):
    print(f"\n--- Stage: {stage_name} ---")
    current_model = pipeline_config.get("model", "gpt2")
    output_dir = pipeline_config.get("output_dir", "./output")
    context_length = pipeline_config.get("context_length", 2048)
    batch_size = pipeline_config.get("batch_size", 4)
    learning_rate = pipeline_config.get("learning_rate", 2e-5)

    stage_output_dir = os.path.join(output_dir, f"model_{stage_name.lower()}")

    dataset_key = f"dataset_{stage_name.lower()}"
    epochs_key = f"epochs_{stage_name.lower()}"

    dataset = stage_config.get(dataset_key, "IlyaGusev/ru_instruct")
    epochs = stage_config.get(epochs_key, 1)

    axolotl_config = create_axolotl_config(
        model_name_or_path=current_model,
        dataset=dataset,
        output_dir=stage_output_dir,
        context_length=context_length,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        is_sft=is_sft
    )

    config_path = os.path.join(output_dir, f"{stage_name.lower()}_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(axolotl_config, f, default_flow_style=False)

    print(f"Generated config at {config_path}")
    run_axolotl(config_path)
    return stage_output_dir

def load_yaml_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml_config(path: str, data: dict):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

@click.command()
@click.option("--config", default="configs/pipeline.yaml", help="Path to main pipeline YAML config.")
def main(config: str):
    """
    LFM-RUS Training Pipeline.
    """
    print("Starting LFM-RUS Training Pipeline")

    pipeline_config = load_yaml_config(config)
    output_dir = pipeline_config.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Base model: {pipeline_config.get('model')}")
    print(f"Output directory: {output_dir}")

    stages = pipeline_config.get("stages", [])

    stage_functions = {
        "pruning": run_pruning,
        "tokenizer": run_tokenizer,
        "warmup": run_warmup,
        "cpt": lambda pc, sc: run_axolotl_stage(pc, sc, "CPT", False),
        "sft": lambda pc, sc: run_axolotl_stage(pc, sc, "SFT", True)
    }

    # config directory
    config_dir = os.path.dirname(config)

    for stage in stages:
        if stage in stage_functions:
            stage_config_path = os.path.join(config_dir, f"{stage}.yaml")
            stage_config = load_yaml_config(stage_config_path)

            # Run stage
            new_model_path = stage_functions[stage](pipeline_config, stage_config)

            # Update and save pipeline config
            pipeline_config["model"] = new_model_path
            save_yaml_config(config, pipeline_config)
        else:
            print(f"Warning: Unknown stage '{stage}' in pipeline configuration.")

    print("\nPipeline completed successfully!")
    print(f"Final model path: {pipeline_config.get('model')}")

if __name__ == "__main__":
    main()
