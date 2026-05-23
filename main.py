import os
import click
import subprocess
import yaml
import sys
from lfm_rus.tokenizer import train_tokenizer
from lfm_rus.lexical_init import initialize_lexical_embeddings

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
    # This assumes axolotl is installed or available in the environment
    cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Axolotl training failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find 'accelerate' or 'axolotl'. Ensure they are installed.")
        # For testing/demonstration purposes where axolotl might not be installed,
        # we'll exit instead of crashing the pipeline silently.
        sys.exit(1)


@click.command()
@click.option("--skip-tokenizer", is_flag=True, help="Skip tokenizer training and lexical initialization.")
@click.option("--skip-cpt", is_flag=True, help="Skip Continual Pre-Training (CPT).")
@click.option("--skip-sft", is_flag=True, help="Skip Supervised Fine-Tuning (SFT).")
@click.option("--model", default="gpt2", help="Base model to use (default: gpt2).")
@click.option("--dataset-tokenizer", default="IlyaGusev/ru_instruct", help="Dataset for tokenizer training.")
@click.option("--dataset-cpt", default="IlyaGusev/ru_instruct", help="Dataset for CPT.")
@click.option("--dataset-sft", default="IlyaGusev/ru_instruct", help="Dataset for SFT.")
@click.option("--context-length", default=2048, type=int, help="Context length for training.")
@click.option("--new-tokens", default=10000, type=int, help="Number of new tokens to add to the tokenizer.")
@click.option("--epochs-cpt", default=1, type=int, help="Number of epochs for CPT.")
@click.option("--epochs-sft", default=1, type=int, help="Number of epochs for SFT.")
@click.option("--batch-size", default=4, type=int, help="Batch size for training.")
@click.option("--learning-rate", default=2e-5, type=float, help="Learning rate for training.")
@click.option("--output-dir", default="./output", help="Output directory for models.")
def main(
    skip_tokenizer: bool,
    skip_cpt: bool,
    skip_sft: bool,
    model: str,
    dataset_tokenizer: str,
    dataset_cpt: str,
    dataset_sft: str,
    context_length: int,
    new_tokens: int,
    epochs_cpt: int,
    epochs_sft: int,
    batch_size: int,
    learning_rate: float,
    output_dir: str
):
    """
    LFM-RUS Training Pipeline.
    """
    print("Starting LFM-RUS Training Pipeline")
    print(f"Base model: {model}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    current_model = model

    # 1. Tokenizer Training
    if not skip_tokenizer:
        print("\n--- Stage 1: Tokenizer Training & Lexical Initialization ---")
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
        current_model = tokenizer_output_dir
    else:
        print("\n--- Skipping Stage 1: Tokenizer Training ---")

    # 2. Continual Pre-Training (CPT)
    if not skip_cpt:
        print("\n--- Stage 2: Continual Pre-Training (CPT) ---")
        cpt_output_dir = os.path.join(output_dir, "model_cpt")
        cpt_config = create_axolotl_config(
            model_name_or_path=current_model,
            dataset=dataset_cpt,
            output_dir=cpt_output_dir,
            context_length=context_length,
            epochs=epochs_cpt,
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_sft=False
        )

        cpt_config_path = os.path.join(output_dir, "cpt_config.yml")
        with open(cpt_config_path, "w") as f:
            yaml.dump(cpt_config, f, default_flow_style=False)

        print(f"Generated CPT config at {cpt_config_path}")
        run_axolotl(cpt_config_path)
        current_model = cpt_output_dir
    else:
        print("\n--- Skipping Stage 2: Continual Pre-Training (CPT) ---")

    # 3. Supervised Fine-Tuning (SFT)
    if not skip_sft:
        print("\n--- Stage 3: Supervised Fine-Tuning (SFT) ---")
        sft_output_dir = os.path.join(output_dir, "model_sft")
        sft_config = create_axolotl_config(
            model_name_or_path=current_model,
            dataset=dataset_sft,
            output_dir=sft_output_dir,
            context_length=context_length,
            epochs=epochs_sft,
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_sft=True
        )

        sft_config_path = os.path.join(output_dir, "sft_config.yml")
        with open(sft_config_path, "w") as f:
            yaml.dump(sft_config, f, default_flow_style=False)

        print(f"Generated SFT config at {sft_config_path}")
        run_axolotl(sft_config_path)
        current_model = sft_output_dir
    else:
        print("\n--- Skipping Stage 3: Supervised Fine-Tuning (SFT) ---")

    print("\nPipeline completed successfully!")
    print(f"Final model path: {current_model}")

if __name__ == "__main__":
    main()
