import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import os

class WarmupDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def embedding_warmup(model_name_or_path: str, texts: list[str], new_tokens: list[str], epochs: int = 1, batch_size: int = 8, lr: float = 1e-3, save_path: str = None):
    """
    Performs embedding warm-up by freezing all layers except new tokens in embeddings.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Make sure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We need to freeze everything, then let ONLY the new tokens get gradients.
    for param in model.parameters():
        param.requires_grad = False

    # We unfreeze all embeddings, BUT we will zero out gradients for old tokens
    # to only train the new ones.
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    for param in input_embeddings.parameters():
        param.requires_grad = True
    for param in output_embeddings.parameters():
        param.requires_grad = True

    dataset = WarmupDataset(tokenizer, texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer (we only pass params that require grad, but AdamW maintains state,
    # we need to be careful with weight decay on zeroed gradients. It's safer to
    # use pure SGD or manually handle Adam, but for simplicity masking grad is fine if weight decay is 0)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Identify the IDs of new tokens
    new_token_ids = []
    for token in new_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(token)
        if tok_id != tokenizer.unk_token_id:
            new_token_ids.append(tok_id)

    # Create a mask for gradients
    vocab_size = input_embeddings.weight.shape[0]
    grad_mask = torch.zeros(vocab_size, 1, device=device)
    if new_token_ids:
        grad_mask[new_token_ids] = 1.0

    # Save original weights to enforce exact old token preservation
    with torch.no_grad():
        orig_in = input_embeddings.weight.clone()
        orig_out = output_embeddings.weight.clone()

    print(f"Starting embedding warmup for {epochs} epochs on device {device}...")
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()

            # Zero out gradients for old tokens before optimizer step
            if input_embeddings.weight.grad is not None:
                input_embeddings.weight.grad *= grad_mask
            if output_embeddings.weight.grad is not None:
                output_embeddings.weight.grad *= grad_mask

            optimizer.step()
            optimizer.zero_grad()

            # Force restore old weights to prevent numerical drift or optimizer state drift
            with torch.no_grad():
                input_embeddings.weight.copy_(torch.where(grad_mask == 1.0, input_embeddings.weight, orig_in))
                output_embeddings.weight.copy_(torch.where(grad_mask == 1.0, output_embeddings.weight, orig_out))

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch+1} average loss: {epoch_loss / len(dataloader):.4f}")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    return model, tokenizer
