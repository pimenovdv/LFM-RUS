import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def prune_tokenizer_and_model(model_name: str, tokens_to_remove: list[str], save_path: str):
    """
    Prunes specific tokens from the tokenizer and model embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    vocab = tokenizer.get_vocab()
    ids_to_remove = {vocab[token] for token in tokens_to_remove if token in vocab}

    if not ids_to_remove:
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
        return model, tokenizer

    vocab_size = len(tokenizer)
    ids_to_keep = [i for i in range(vocab_size) if i not in ids_to_remove]

    # Resize model embeddings
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        new_input_embeddings = torch.nn.Embedding(len(ids_to_keep), input_embeddings.embedding_dim)
        new_input_embeddings.weight.data = input_embeddings.weight.data[ids_to_keep]
        model.set_input_embeddings(new_input_embeddings)

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        new_output_embeddings = torch.nn.Linear(output_embeddings.in_features, len(ids_to_keep), bias=False)
        new_output_embeddings.weight.data = output_embeddings.weight.data[ids_to_keep]
        model.set_output_embeddings(new_output_embeddings)

    model.config.vocab_size = len(ids_to_keep)

    old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(ids_to_keep)}

    # Fix special tokens
    if model.config.bos_token_id is not None:
        model.config.bos_token_id = old_to_new_id.get(model.config.bos_token_id, model.config.bos_token_id)
    if model.config.eos_token_id is not None:
        model.config.eos_token_id = old_to_new_id.get(model.config.eos_token_id, model.config.eos_token_id)
    if model.config.pad_token_id is not None:
        model.config.pad_token_id = old_to_new_id.get(model.config.pad_token_id, model.config.pad_token_id)


    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        tokenizer_json_path = os.path.join(save_path, "tokenizer.json")
        vocab_path = os.path.join(save_path, "vocab.json")
        merges_path = os.path.join(save_path, "merges.txt")

        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)

            if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                new_vocab = {}
                for token, old_id in tokenizer_data["model"]["vocab"].items():
                    if old_id not in ids_to_remove:
                        new_vocab[token] = old_to_new_id[old_id]
                tokenizer_data["model"]["vocab"] = new_vocab

            if "added_tokens" in tokenizer_data:
                new_added_tokens = []
                for token_obj in tokenizer_data["added_tokens"]:
                    if token_obj["id"] not in ids_to_remove:
                        token_obj["id"] = old_to_new_id.get(token_obj["id"], token_obj["id"])
                        new_added_tokens.append(token_obj)
                tokenizer_data["added_tokens"] = new_added_tokens

            if "model" in tokenizer_data and "merges" in tokenizer_data["model"]:
                new_merges = []
                for merge in tokenizer_data["model"]["merges"]:
                    if isinstance(merge, str):
                        parts = merge.split()
                    else:
                        parts = merge
                    if len(parts) == 2:
                        merged = "".join(parts)
                        if merged in tokens_to_remove or any(rt in merged for rt in tokens_to_remove):
                            continue
                    new_merges.append(merge)
                tokenizer_data["model"]["merges"] = new_merges

            with open(tokenizer_json_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            new_vocab = {}
            for token, old_id in vocab_data.items():
                if old_id not in ids_to_remove:
                    new_vocab[token] = old_to_new_id[old_id]
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(new_vocab, f, ensure_ascii=False, indent=2)

        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                merges_lines = f.readlines()

            new_merges = []
            for line in merges_lines:
                if line.startswith("#"):
                    new_merges.append(line)
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    merged = "".join(parts)
                    if merged in tokens_to_remove or any(rt in merged for rt in tokens_to_remove):
                        continue
                new_merges.append(line)

            with open(merges_path, "w", encoding="utf-8") as f:
                f.writelines(new_merges)

        tokenizer_config_path = os.path.join(save_path, "tokenizer_config.json")
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

        # Reload
        tokenizer = AutoTokenizer.from_pretrained(save_path)

    return model, tokenizer
