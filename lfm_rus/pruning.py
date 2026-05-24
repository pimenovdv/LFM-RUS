import json
import os
import torch
import torch.nn as nn
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import tempfile

def prune_tokenizer_and_model(
    model_name: str,
    datasets: list[dict],
    save_path: str,
    min_freq: int = 100,
    max_samples: int = 500000
):
    """
    Prunes specific tokens from the tokenizer and model embeddings based on dataset frequency.
    """
    print(f"1. Loading model and tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("2. Counting token frequencies...")
    token_counts = Counter()
    for ds_config in datasets:
        ds_path = ds_config["path"]
        ds_name = ds_config.get("name")
        print(f"Loading dataset: {ds_path} (name={ds_name})")
        if ds_name:
            try:
                dataset = load_dataset(ds_path, name=ds_name, split="train", streaming=True, **{k:v for k,v in ds_config.items() if k not in ["path", "name"]})
            except ValueError:
                dataset = load_dataset(ds_path, name=ds_name, split="test", streaming=True, **{k:v for k,v in ds_config.items() if k not in ["path", "name"]})
            except Exception:
                dataset = load_dataset(ds_path, name=ds_name, streaming=True, **{k:v for k,v in ds_config.items() if k not in ["path", "name"]})
        else:
            try:
                dataset = load_dataset(ds_path, split="train", streaming=True, **{k:v for k,v in ds_config.items() if k not in ["path", "name"]})
            except ValueError:
                dataset = load_dataset(ds_path, split="test", streaming=True, **{k:v for k,v in ds_config.items() if k not in ["path", "name"]})
            except Exception:
                dataset = load_dataset(ds_path, streaming=True, **{k:v for k,v in ds_config.items() if k not in ["path", "name"]})

        subset = dataset.take(max_samples)

        for item in tqdm(subset, desc=f"Processing {ds_path}"):
            text = item.get("text", "")
            if text:
                ids = tokenizer.encode(text, add_special_tokens=False)
                token_counts.update(ids)

    print("3. Selecting tokens to keep...")
    keep_ids = []
    special_tokens_ids = set(tokenizer.all_special_ids)

    vocab_size = len(tokenizer)
    for old_id in range(vocab_size):
        if old_id in special_tokens_ids or token_counts[old_id] >= min_freq:
            keep_ids.append(old_id)

    new_vocab_size = len(keep_ids)
    print(f"Old vocab size: {vocab_size}")
    print(f"New vocab size: {new_vocab_size}")
    print(f"Tokens to be removed: {vocab_size - new_vocab_size}")

    if vocab_size == new_vocab_size:
        print("No tokens to prune based on the given min_freq.")
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
        return model, tokenizer

    print("4. Updating tokenizer (JSON)...")
    old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
    ids_to_remove = set(range(vocab_size)) - set(keep_ids)

    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer.save_pretrained(temp_dir)

        tokenizer_json_path = os.path.join(temp_dir, "tokenizer.json")
        tok_data = {}
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                tok_data = json.load(f)

        if "model" in tok_data and "vocab" in tok_data["model"]:
            old_vocab = tok_data["model"]["vocab"]
            id_to_tok_str = {v: k for k, v in old_vocab.items()}

            new_vocab = {}
            for old_id in keep_ids:
                tok_str = id_to_tok_str.get(old_id)
                if tok_str is not None:
                    new_vocab[tok_str] = old_to_new_id[old_id]

            tok_data["model"]["vocab"] = new_vocab

        if "added_tokens" in tok_data:
            new_added_tokens = []
            for token_obj in tok_data["added_tokens"]:
                if token_obj["id"] not in ids_to_remove:
                    token_obj["id"] = old_to_new_id.get(token_obj["id"], token_obj["id"])
                    new_added_tokens.append(token_obj)
            tok_data["added_tokens"] = new_added_tokens

        if "model" in tok_data and "merges" in tok_data["model"]:
            new_merges = []
            valid_chars = set(new_vocab.keys())
            for merge in tok_data["model"]["merges"]:
                if isinstance(merge, str):
                    parts = merge.split()
                else:
                    parts = merge
                if len(parts) == 2:
                    if parts[0] in valid_chars and parts[1] in valid_chars:
                        merged = "".join(parts)
                        if merged in valid_chars:
                            new_merges.append(merge)
            tok_data["model"]["merges"] = new_merges

        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "w", encoding="utf-8") as f:
                json.dump(tok_data, f, ensure_ascii=False, indent=2)

        vocab_path = os.path.join(temp_dir, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(new_vocab, f, ensure_ascii=False, indent=2)

        merges_path = os.path.join(temp_dir, "merges.txt")
        if os.path.exists(merges_path) and "model" in tok_data and "merges" in tok_data["model"]:
            with open(merges_path, "w", encoding="utf-8") as f:
                f.write("#version: 0.2\n")
                for merge in tok_data["model"]["merges"]:
                    if isinstance(merge, list):
                        merge = " ".join(merge)
                    f.write(merge + "\n")


        tokenizer_config_path = os.path.join(temp_dir, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            if "added_tokens_decoder" in config_data:
                new_decoder = {}
                for old_id_str, token_obj in config_data["added_tokens_decoder"].items():
                    old_id_int = int(old_id_str)
                    if old_id_int not in ids_to_remove:
                        new_id = old_to_new_id.get(old_id_int, old_id_int)
                        new_decoder[str(new_id)] = token_obj
                config_data["added_tokens_decoder"] = new_decoder
            with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

        pruned_tokenizer = AutoTokenizer.from_pretrained(temp_dir)


    print("5. Updating model matrices (PyTorch)...")
    keep_indices_tensor = torch.tensor(keep_ids, dtype=torch.long)

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        old_embeddings = input_embeddings.weight.data
        new_embeddings_data = old_embeddings[keep_indices_tensor]

        new_embeddings = nn.Embedding(new_vocab_size, input_embeddings.embedding_dim, dtype=input_embeddings.weight.dtype)
        new_embeddings.weight.data = new_embeddings_data
        model.set_input_embeddings(new_embeddings)

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        old_lm_head = output_embeddings.weight.data
        new_lm_head_data = old_lm_head[keep_indices_tensor]

        new_lm_head = nn.Linear(output_embeddings.in_features, new_vocab_size, bias=False, dtype=output_embeddings.weight.dtype)
        new_lm_head.weight.data = new_lm_head_data
        model.set_output_embeddings(new_lm_head)

    model.config.vocab_size = new_vocab_size

    if model.config.bos_token_id is not None:
        model.config.bos_token_id = old_to_new_id.get(model.config.bos_token_id, model.config.bos_token_id)
    if model.config.eos_token_id is not None:
        model.config.eos_token_id = old_to_new_id.get(model.config.eos_token_id, model.config.eos_token_id)
    if model.config.pad_token_id is not None:
        model.config.pad_token_id = old_to_new_id.get(model.config.pad_token_id, model.config.pad_token_id)

    if save_path:
        print("6. Saving final model...")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        pruned_tokenizer.save_pretrained(save_path)

    return model, pruned_tokenizer
