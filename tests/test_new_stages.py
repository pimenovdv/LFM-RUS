import pytest
import os
import torch
from lfm_rus.pruning import prune_tokenizer_and_model
from lfm_rus.embedding_warmup import embedding_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile

def test_pruning():
    # Use a small fast model
    model_name = "gpt2"
    with tempfile.TemporaryDirectory() as tmpdir:
        # Before pruning
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_size = len(base_tokenizer)

        # We will prune the word 'hello'
        tokens_to_prune = ["hello"]

        model, tokenizer = prune_tokenizer_and_model(
            model_name=model_name,
            tokens_to_remove=tokens_to_prune,
            save_path=tmpdir
        )

        # Verify vocab size
        assert len(tokenizer) == base_size - 1
        assert "hello" not in tokenizer.get_vocab()

        # Verify model embeddings
        assert model.get_input_embeddings().weight.shape[0] == base_size - 1

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
