import pytest
import os
import yaml
import tempfile
import shutil
from unittest.mock import patch, MagicMock

@pytest.fixture
def small_pipeline_config(tmp_path):
    output_dir = tmp_path / "output"
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # We create a dummy dataset locally
    dummy_text_path = tmp_path / "dummy.jsonl"
    with open(dummy_text_path, "w") as f:
        for _ in range(10):
            f.write('{"text": "Привет мир! Это тестовое предложение для маленькой модели."}\n')

    # Mocking HF datasets for local files requires proper configuration.
    # For a simple test, using 'json' and 'data_files' works but defaults to train split usually if configured properly.

    # Very small model for testing
    model_name = "sshleifer/tiny-gpt2"

    pipeline_yaml = {
        "model": model_name,
        "output_dir": str(output_dir),
        "configs_dir": str(configs_dir),
        "max_len": 32,
        "push_to_hub": False,
        "stages": ["pruning", "tokenizer", "warmup", "cpt", "sft"]
    }

    pipeline_file = tmp_path / "pipeline.yaml"
    with open(pipeline_file, "w") as f:
        yaml.dump(pipeline_yaml, f)

    dataset_config = [{"path": "json", "data_files": str(dummy_text_path), "split": "train"}]

    pruning_yaml = {
        "datasets": dataset_config,
        "min_freq": 1,
        "max_samples": 10
    }
    with open(configs_dir / "pruning.yaml", "w") as f:
        yaml.dump(pruning_yaml, f)

    tokenizer_yaml = {
        "dataset_tokenizer": dataset_config,
        "new_tokens": 10
    }
    with open(configs_dir / "tokenizer.yaml", "w") as f:
        yaml.dump(tokenizer_yaml, f)

    warmup_yaml = {
        "dataset_warmup": dataset_config,
        "epochs_warmup": 1,
        "batch_size": 2,
        "learning_rate": 1e-3
    }
    with open(configs_dir / "warmup.yaml", "w") as f:
        yaml.dump(warmup_yaml, f)

    cpt_yaml = {
        "datasets": dataset_config,
    }
    with open(configs_dir / "cpt.yaml", "w") as f:
        yaml.dump(cpt_yaml, f)

    sft_yaml = {
        "datasets": dataset_config,
    }
    with open(configs_dir / "sft.yaml", "w") as f:
        yaml.dump(sft_yaml, f)

    return pipeline_file

@patch("main.run_axolotl")

@patch("main.initialize_lexical_embeddings")
def test_full_pipeline_with_tiny_model(mock_lexical, mock_run_axolotl, small_pipeline_config):
    from main import main
    from click.testing import CliRunner

    def fake_run_axolotl(config_path):
        import shutil
        import yaml

        # Simulate axolotl creating the output directory and a dummy config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        output_dir = cfg["output_dir"]
        base_model = cfg["base_model"]

        # Instead of training, we just copy the base model to the output directory
        # to ensure the next stages can load it.
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(base_model, output_dir)

    mock_run_axolotl.side_effect = fake_run_axolotl

    def fake_lexical(model_name, new_tokens, save_path):
        import os
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Don't bother resizing here, mock just needs to save something
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        return model, tokenizer

    mock_lexical.side_effect = fake_lexical




    def fake_load_dataset(*args, **kwargs):
        class MockDataset:
            def __init__(self):
                pass
            def take(self, n):
                return [{"text": "Привет мир! Это тестовое предложение для маленькой модели."}] * n
        return MockDataset()

    with patch("lfm_rus.tokenizer.load_dataset", side_effect=fake_load_dataset), \
         patch("lfm_rus.pruning.load_dataset", side_effect=fake_load_dataset), \
         patch("main.load_dataset", side_effect=fake_load_dataset):
        runner = CliRunner()
        result = runner.invoke(main, ["--config", str(small_pipeline_config)])


    if result.exit_code != 0:
        print(result.output)
        print(result.exception)

    assert result.exit_code == 0
    assert mock_run_axolotl.call_count == 2
