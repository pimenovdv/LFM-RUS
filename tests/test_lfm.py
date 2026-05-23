import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lfm_rus.tokenizer import train_tokenizer, get_training_corpus
from lfm_rus.lexical_init import initialize_lexical_embeddings
from unittest.mock import patch, MagicMock

def test_get_training_corpus():
    # Mock dataset
    mock_dataset = [{"text": "text1"}, {"text": "text2"}, {"text": "text3"}]
    generator = get_training_corpus(mock_dataset, batch_size=2)

    batch1 = next(generator)
    assert batch1 == ["text1", "text2"]

    batch2 = next(generator)
    assert batch2 == ["text3"]

    with pytest.raises(StopIteration):
        next(generator)

@patch("lfm_rus.tokenizer.load_dataset")
def test_train_tokenizer(mock_load_dataset):
    # Create a dummy dataset
    class DummyDataset:
        def __init__(self, data):
            self.data = data
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return self.data[idx]
            return self.data[idx]
        def __len__(self):
            return len(self.data)

    # Russian sentence that gpt2 shouldn't have tokens for by default
    dummy_data = DummyDataset([{"text": "Кошка сидит на столе"} for _ in range(10)])
    mock_load_dataset.return_value = dummy_data

    added_tokens, tokenizer = train_tokenizer(
        base_model_name="gpt2",
        dataset_name="dummy",
        vocab_size=50257,
        new_tokens=5
    )

    # Assert that tokenizer has new tokens
    assert len(added_tokens) > 0
    # Use baseline gpt2 size
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert len(tokenizer) > len(base_tokenizer)

@patch("lfm_rus.lexical_init.MarianMTModel")
@patch("lfm_rus.lexical_init.MarianTokenizer")
def test_initialize_lexical_embeddings(mock_marian_tokenizer_class, mock_marian_model_class):
    # Mock translation
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_marian_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    mock_marian_model_class.from_pretrained.return_value = mock_model

    # When translating, just return a dummy tensor so trans_tokenizer.decode is called
    mock_model.generate.return_value = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    mock_tokenizer.decode.return_value = "cat"
    mock_tokenizer.side_effect = lambda x, **kw: {"input_ids": x}

    # Check what happens with gpt2 and a couple of fake tokens
    # Using small model
    new_tokens = ["Ġкошка", "Ġсобака", "ство"]

    # Run the function
    model, tokenizer = initialize_lexical_embeddings(
        model_name="gpt2",
        new_tokens=new_tokens,
        save_path=None # don't save
    )

    # Original vocab size for gpt2 is 50257
    assert len(tokenizer) == 50257 + 3

    # Check that model embeddings have been resized
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    assert input_embeddings.shape[0] == 50257 + 3
    assert output_embeddings.shape[0] == 50257 + 3

    # Ensure weights are not zero for the new tokens
    for token in new_tokens:
        idx = tokenizer.convert_tokens_to_ids(token)
        assert not torch.all(input_embeddings[idx] == 0)
        assert not torch.all(output_embeddings[idx] == 0)
