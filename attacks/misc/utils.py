"""
Utility functions for membership inference attacks.

Provides common functionality for computing losses, loading reference models,
and creating few-shot prefixes for recall-based attacks.
"""
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import login
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import init_model


def compute_nlloss(
        model: PreTrainedModel,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ignore_prefix: Optional[int] = None,
):
    """
    Compute negative log-likelihood loss.

    Args:
        model: The language model
        token_ids: Input token IDs
        attention_mask: Attention mask
        ignore_prefix: Number of prefix tokens to ignore

    Returns:
        Per-sample loss array
    """
    with torch.no_grad():
        labels = token_ids.clone()

        outputs = model(token_ids, attention_mask=attention_mask)

        # Handle DataParallel output
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if isinstance(model, torch.nn.DataParallel):
            vocab_size = model.module.config.vocab_size
        else:
            vocab_size = model.config.vocab_size

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
        loss = loss.view(token_ids.shape[0], -1)

        if ignore_prefix:
            loss = loss[:, ignore_prefix:]
            shift_attention_mask = shift_attention_mask[:, ignore_prefix:]

        loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)

        return loss.detach().cpu().numpy()


def batch_nlloss(batch, model, tokenizer, device, key='nlloss', max_length=512):
    """
    Compute batch negative log-likelihood loss.

    Args:
        batch: Input batch
        model: The language model
        tokenizer: Tokenizer
        device: Device to run on
        key: Key to store results
        max_length: Maximum sequence length

    Returns:
        Dictionary with loss values
    """
    texts = batch['text']
    tokenized = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True,
                                            truncation=True, max_length=max_length)
    token_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    # Split the batch into smaller chunks
    chunk_size = 1  # Process one sample at a time
    losses = []
    for i in range(0, len(texts), chunk_size):
        chunk_ids = token_ids[i:i + chunk_size]
        chunk_mask = attention_mask[i:i + chunk_size]
        chunk_losses = compute_nlloss(model, chunk_ids, chunk_mask)
        losses.extend(chunk_losses)

    return {key: losses}


def compute_per_token_losses(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        texts: list,
        device: torch.device,
        max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token losses.

    Args:
        model: The language model
        tokenizer: Tokenizer
        texts: List of input texts
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        Tuple of (per_token_losses, valid_mask)
    """
    model.eval()

    # Handle tokenizer padding
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    # Tokenize
    tokenized = tokenizer.batch_encode_plus(
        texts, return_tensors='pt', padding='max_length', truncation=True,
        max_length=max_length, return_attention_mask=True
    )
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = getattr(outputs, "logits", outputs[0])

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attention = attention_mask[..., 1:].contiguous()

        # Mask invalid tokens
        shift_labels[shift_attention == 0] = -100

        # Compute per-token loss
        if isinstance(model, torch.nn.DataParallel):
            vocab_size = model.module.config.vocab_size
        else:
            vocab_size = model.config.vocab_size

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, vocab_size), shift_labels.view(-1),
            reduction='none', ignore_index=-100
        ).view(input_ids.shape[0], -1)

        valid_mask = (shift_labels != -100)

        return loss_per_token.cpu().numpy(), valid_mask.cpu().numpy()


def make_recall_prefix(dataset, n_shots, perplexity_bucket=None):
    """Create a prefix from random samples in the dataset."""
    if perplexity_bucket is not None:
        dataset = dataset.filter(lambda x: x["perplexity_bucket"] == perplexity_bucket)

    indices = random.sample(range(len(dataset)), n_shots)
    prefixes = [dataset[i]["text"] for i in indices]

    return " ".join(prefixes)


def load_reference(
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = 'cuda',
        hf_token: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from the given paths and move them to the specified device.

    Args:
        model_path (str): Path or identifier for the model.
        tokenizer_path (Optional[str]): Path or identifier for the tokenizer. Defaults to model_path.
        device (str): Device to load the model on. Defaults to 'cuda'.
        hf_token (Optional[str]): Hugging Face token for authentication.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Loaded model and tokenizer.
    """
    if tokenizer_path is None:
        tokenizer_path = model_path
    if hf_token:
        login(token=hf_token)

    model, tokenizer, _ = init_model(model_name=model_path, tokenizer_name=tokenizer_path, device=device)
    model.eval()
    return model, tokenizer


def unload_reference(
        model: Optional[PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizer],
        device: torch.device
) -> None:
    """
    Unload the model and tokenizer to free memory.

    Args:
        model (Optional[PreTrainedModel]): The model to unload.
        tokenizer (Optional[PreTrainedTokenizer]): The tokenizer to unload.
        device (torch.device): The device the model was loaded on.
    """
    if model is not None:
        if hasattr(model, 'to'):
            model.to('cpu')
        del model
    if tokenizer is not None:
        del tokenizer
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.empty_cache()


def _get_model_base_config(model_instance: PreTrainedModel):
    if isinstance(model_instance, torch.nn.DataParallel):
        return model_instance.module.config
    return model_instance.config
