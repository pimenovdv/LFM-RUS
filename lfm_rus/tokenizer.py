import json
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
from typing import List, Generator, Tuple, Union

def get_training_corpus(dataset, batch_size: int = 1000) -> Generator[List[str], None, None]:
    for i in range(0, len(dataset), batch_size):
        yield [item["text"] for item in dataset[i : i + batch_size]]

def train_tokenizer(base_model_name: str, dataset_name: str, vocab_size: int = 50257, new_tokens: int = 10000) -> Tuple[List[str], Union[AutoTokenizer, PreTrainedTokenizerFast]]:
    """
    Trains a new tokenizer from a base tokenizer on a specific dataset.
    Returns the list of newly added tokens.
    """
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load dataset. E.g., IlyaGusev/ru_instruct or wikipedia
    # Note: We use a small portion for demonstration/fast training if this runs in test
    dataset = load_dataset(dataset_name, split="train")

    # create generator
    training_corpus = get_training_corpus(dataset)

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
    added, tokenizer = train_tokenizer("gpt2", "IlyaGusev/ru_instruct", vocab_size=50257, new_tokens=5000)
    print(f"Added {len(added)} tokens. Sample: {added[:10]}")
