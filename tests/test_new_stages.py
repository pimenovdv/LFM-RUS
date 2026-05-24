import pytest
import os
import torch
from unittest.mock import patch
from datasets import Dataset
from lfm_rus.pruning import prune_tokenizer_and_model
from lfm_rus.embedding_warmup import embedding_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile

@patch('lfm_rus.pruning.load_dataset')
def test_pruning(mock_load_dataset):
    # Setup dummy dataset
    # "apple" appears 1 time, "banana" appears 2 times, "hello" appears 3 times
    # In GPT2, " apple" (or similar), depends on the tokenization
    # Let's just create some text. We will set min_freq=2, so rare words are dropped

    # Let's use words that are single tokens in gpt2
    # 'hello': 31373
    # 'world': 6894
    # 'apple': 17180

    # Let's just pass text and see what gets pruned.
    # We'll make it so "apple" has freq 1, "hello" has freq 2.
    dummy_data = {
        "text": ["hello world apple", "hello world banana", "world world world"]
    }
    dummy_dataset = Dataset.from_dict(dummy_data)
    mock_load_dataset.return_value = dummy_dataset

    model_name = "gpt2"
    with tempfile.TemporaryDirectory() as tmpdir:
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_size = len(base_tokenizer)

        # We will prune tokens that appear less than 2 times
        # Wait, if we use a small dataset, ALL OTHER TOKENS in vocab (50k) will have freq 0 and be dropped.
        # This will test pruning very well, reducing vocab size drastically!

        model, tokenizer = prune_tokenizer_and_model(
            model_name=model_name,
            datasets=["dummy"],
            min_freq=1,
            max_samples=10,
            save_path=tmpdir
        )

        # New vocab should be extremely small (special tokens + tokens in our dummy data)
        assert len(tokenizer) < 1000
        assert len(tokenizer) > 0

        # Model embeddings should match the new small vocab
        assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)

def test_embedding_warmup():
    model_name = "gpt2"
    with tempfile.TemporaryDirectory() as tmpdir:
        # Basic texts
        texts = ["Hello world", "This is a test"]

        # Warmup
        model, tokenizer = embedding_warmup(
            model_name_or_path=model_name,
            texts=texts,
            new_tokens=[],
            epochs=1,
            batch_size=1,
            save_path=tmpdir
        )

        # Ensure it works without errors and saves
        assert os.path.exists(os.path.join(tmpdir, "config.json"))

        # Check that only embeddings are trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        # Embeddings + lm_head should be trainable, others frozen
        assert trainable_params < total_params
        assert trainable_params > 0
