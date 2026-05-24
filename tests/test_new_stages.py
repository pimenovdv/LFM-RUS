import pytest
import os
import torch
import tempfile
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer, AutoModelForCausalLM
from lfm_rus.pruning import prune_tokenizer_and_model
from lfm_rus.embedding_warmup import embedding_warmup
from datasets import Dataset

def test_embedding_warmup():
    model_name = "gpt2"
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Add some fake tokens to gpt2 and save it to tempdir
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # ensure pad token exists for training
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        new_tokens = ["Ġкошка", "Ġсобака"]
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))

        # Create some original copies to test change
        original_embeddings = model.get_input_embeddings().weight.data.clone()

        model.save_pretrained(tmpdir)
        tokenizer.save_pretrained(tmpdir)

        texts = ["кошка сидит", "собака бежит", "кошка собака"]

        # Run warmup
        embedding_warmup(
            model_name_or_path=tmpdir,
            texts=texts,
            new_tokens=new_tokens,
            epochs=1,
            batch_size=2,
            lr=1e-3,
            save_path=tmpdir
        )

        # Load back and check that new tokens changed, but old tokens did not
        new_model = AutoModelForCausalLM.from_pretrained(tmpdir)
        new_embeddings = new_model.get_input_embeddings().weight.data

        # Check old token (e.g., "hello" id 31373)
        assert torch.allclose(original_embeddings[31373], new_embeddings[31373]), "Old embeddings should not change during warmup"

        # Check new token
        idx_cat = tokenizer.convert_tokens_to_ids("Ġкошка")
        assert not torch.allclose(original_embeddings[idx_cat], new_embeddings[idx_cat]), "New embeddings should change during warmup"

@patch('lfm_rus.pruning.load_dataset')
def test_pruning(mock_load_dataset):
    dummy_data = {
        "text": ["hello world apple", "hello world banana", "world world world"]
    }
    class MockDataset:
        def take(self, n):
            return [{"text": t} for t in dummy_data["text"][:n]]

    mock_load_dataset.return_value = MockDataset()

    model_name = "gpt2"
    with tempfile.TemporaryDirectory() as tmpdir:
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_size = len(base_tokenizer)

        model, tokenizer = prune_tokenizer_and_model(
            model_name=model_name,
            datasets=[{"path": "dummy"}],
            min_freq=1,
            max_samples=10,
            save_path=tmpdir
        )

        new_size = len(tokenizer)
        assert new_size < base_size, "Vocab should be pruned"
        assert model.get_input_embeddings().weight.shape[0] == new_size
        assert model.get_output_embeddings().weight.shape[0] == new_size
