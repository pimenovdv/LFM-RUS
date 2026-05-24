import json
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
from typing import List, Generator, Tuple, Union

def get_training_corpus(datasets_list: List[dict], max_samples: int = 50000) -> Generator[List[str], None, None]:
    for ds_config in datasets_list:
        ds_path = ds_config["path"]
        ds_name = ds_config.get("name")
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

        # yield chunks of texts
        batch_size = 1000
        batch = []
        for item in subset:
            batch.append(item["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def train_tokenizer(base_model_name: str, datasets: List[dict], vocab_size: int = 50257, new_tokens: int = 10000) -> Tuple[List[str], Union[AutoTokenizer, PreTrainedTokenizerFast]]:
    """
    Trains a new tokenizer from a base tokenizer on a specific dataset.
    Returns the list of newly added tokens.
    """
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # create generator
    training_corpus = get_training_corpus(datasets)

    # Check original vocab size to handle models like gpt2 properly
    orig_vocab_size = len(base_tokenizer)

    # train the tokenizer
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        training_corpus,
        vocab_size=orig_vocab_size + new_tokens
    )

    # Fast tokenizers often start from scratch and have smaller len if trained from small iterators.
    # We should merge the new tokens into the original tokenizer instead.

    # Find new tokens by comparing vocabularies
    old_vocab = set(base_tokenizer.get_vocab().keys())
    new_vocab = set(new_tokenizer.get_vocab().keys())

    added_tokens = list(new_vocab - old_vocab)

    # Add new tokens to base tokenizer
    base_tokenizer.add_tokens(added_tokens)

    return added_tokens, base_tokenizer

if __name__ == "__main__":
    # Example usage
    added, tokenizer = train_tokenizer("gpt2", [{"path": "IlyaGusev/ru_instruct"}], vocab_size=50257, new_tokens=5000)
    print(f"Added {len(added)} tokens. Sample: {added[:10]}")
