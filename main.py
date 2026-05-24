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

def get_warmup_texts(datasets_list, max_samples=10000):
    texts = []
    for ds_config in datasets_list:
        try:
            ds_path = ds_config["path"]
            ds_name = ds_config.get("name")
            if ds_name:
                dataset = load_dataset(ds_path, name=ds_name, split="train", streaming=True)
            else:
                dataset = load_dataset(ds_path, split="train", streaming=True)

            subset = dataset.take(max_samples)
            texts.extend([item["text"] for item in subset])
        except Exception as e:
            print(f"Error loading dataset {ds_config} for warmup: {e}")
    return texts

@click.command()
@click.option("--config", default="pipeline.yaml", help="Path to pipeline configuration YAML file.")
def main(config: str):
    """
    LFM-RUS Training Pipeline.
    """
    print(f"Starting LFM-RUS Training Pipeline using config: {config}")

    with open(config, "r") as f:
        pipeline_config = yaml.safe_load(f)

    model = pipeline_config.get("model", "LiquidAI/LFM2.5-350M")
    output_dir = pipeline_config.get("output_dir", "./output")
    max_len = pipeline_config.get("max_len", 2048)
    stages = pipeline_config.get("stages", [])

    print(f"Base model: {model}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    current_model = model
    added_tokens = []

    for stage in stages:
        stage_config_path = f"configs/{stage}.yaml"
        if os.path.exists(stage_config_path):
            with open(stage_config_path, "r") as f:
                stage_config = yaml.safe_load(f)
        else:
            print(f"Warning: Config file {stage_config_path} not found. Using empty config.")
            stage_config = {}

        # Merge pipeline_config into stage_config for flexibility, but give priority to stage_config if exists
        for key, value in pipeline_config.items():
            if key not in ["model", "stages", "output_dir", "max_len"]:
                if key not in stage_config:
                    stage_config[key] = value

        if stage == "pruning":
            print("\n--- Stage 0: Pruning ---")
            datasets = stage_config.get("datasets", [])
            min_freq = stage_config.get("min_freq", 100)
            max_samples = stage_config.get("max_samples", 500000)
            if datasets:
                print(f"Pruning using datasets: {datasets}")
                prune_output_dir = os.path.join(output_dir, "model_pruned")
                prune_tokenizer_and_model(
                    model_name=current_model,
                    datasets=datasets,
                    min_freq=min_freq,
                    max_samples=max_samples,
                    save_path=prune_output_dir
                )
                current_model = prune_output_dir
            else:
                print("No datasets for pruning provided. Skipping pruning.")

        elif stage == "tokenizer":
            print("\n--- Stage 1: Tokenizer Training & Lexical Initialization ---")
            dataset_tokenizer = stage_config.get("dataset_tokenizer", [{"path": "IlyaGusev/ru_instruct"}])
            new_tokens = stage_config.get("new_tokens", 10000)

            print(f"Training tokenizer on dataset: {dataset_tokenizer}")
            added_tokens, tokenizer = train_tokenizer(
                base_model_name=current_model,
                datasets=dataset_tokenizer,
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

        elif stage == "warmup":
            print("\n--- Stage 1.5: Embedding Warm-up ---")
            dataset_warmup = stage_config.get("dataset_warmup", [{"path": "IlyaGusev/ru_instruct"}])
            epochs_warmup = stage_config.get("epochs_warmup", 1)
            batch_size = stage_config.get("batch_size", 4)
            learning_rate = stage_config.get("learning_rate", 2e-5)

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
                current_model = warmup_output_dir
            else:
                print("No texts found for warmup. Skipping Embedding Warm-up.")

        elif stage == "cpt":
            print("\n--- Stage 2: Continual Pre-Training (CPT) ---")
            cpt_output_dir = os.path.join(output_dir, "model_cpt")

            # Inject global pipeline settings
            stage_config["base_model"] = current_model
            stage_config["output_dir"] = cpt_output_dir
            stage_config["sequence_len"] = max_len
            if "dataset_prepared_path" not in stage_config or stage_config["dataset_prepared_path"] == "./data_prepared_cpt":
                stage_config["dataset_prepared_path"] = os.path.join(cpt_output_dir, "data_prepared")

            cpt_config_path = os.path.join(output_dir, "cpt_config.yml")
            with open(cpt_config_path, "w") as f:
                yaml.dump(stage_config, f, default_flow_style=False)

            print(f"Generated CPT config at {cpt_config_path}")
            run_axolotl(cpt_config_path)
            current_model = cpt_output_dir

        elif stage == "sft":
            print("\n--- Stage 3: Supervised Fine-Tuning (SFT) ---")
            sft_output_dir = os.path.join(output_dir, "model_sft")

            # Inject global pipeline settings
            stage_config["base_model"] = current_model
            stage_config["output_dir"] = sft_output_dir
            stage_config["sequence_len"] = max_len
            if "dataset_prepared_path" not in stage_config:
                stage_config["dataset_prepared_path"] = os.path.join(sft_output_dir, "data_prepared")

            sft_config_path = os.path.join(output_dir, "sft_config.yml")
            with open(sft_config_path, "w") as f:
                yaml.dump(stage_config, f, default_flow_style=False)

            print(f"Generated SFT config at {sft_config_path}")
            run_axolotl(sft_config_path)
            current_model = sft_output_dir

    print("\nPipeline completed successfully!")
    print(f"Final model path: {current_model}")

if __name__ == "__main__":
    main()
