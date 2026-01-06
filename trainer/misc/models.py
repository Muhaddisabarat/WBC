import json
import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM

from misc.utils import format_param_count


logger = logging.getLogger(__name__)

def load_model(
    model_identifier: str,
    load_pretrained: bool,
    checkpoint_path: Optional[str] = None
):
    """
    Loads a model.

    Args:
        model_identifier (str):
            - Path to a local JSON config (e.g. './configs/my_config.json'), or
            - A HF Hub identifier (e.g. 'allenai/OLMoE-1B-7B-0924').
        load_pretrained (bool):
            If False, the "from_config" logic is used:
               - If model_identifier ends with .json => interpret as local config => from_config
               - Else => from_config with HF Hub config.
            If True, load pretrained weights:
               - If checkpoint_path is provided => from_pretrained(checkpoint_path)
               - Otherwise => from_pretrained(model_identifier).
        checkpoint_path (str, optional):
            Local path to a directory containing a saved (fine-tuned) model checkpoint.
            Used only if load_pretrained=True.

    Returns:
        model: The loaded model.
    """
    torch_dtype = torch.bfloat16
    trust_remote_code = True

    if load_pretrained:
        # Use pretrained weights.
        source = checkpoint_path if checkpoint_path is not None else model_identifier
        ModelClass = AutoModelForCausalLM # TODO: we can handle this later
        model = ModelClass.from_pretrained(
            source,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if checkpoint_path is not None:
            logger.info(f"Loaded model from local checkpoint: {checkpoint_path}")
        else:
            logger.info(f"Loaded model from HF Hub: {model_identifier}")
    else:
        # Load configuration-only and construct the model.
        if model_identifier.endswith(".json"):
            with open(model_identifier, "r") as f:
                config_dict = json.load(f)
            config = AutoConfig.from_dict(config_dict)
            logger.info(f"Loaded model configuration from local JSON: {model_identifier}")
        else:
            config = AutoConfig.from_pretrained(
                model_identifier,
                trust_remote_code=trust_remote_code,
            )
            logger.info(f"Loaded model configuration from HF Hub: {model_identifier}")
        # Force local construction by disabling remote code loading.
        model = AutoModel.from_config(config, trust_remote_code=False)
        logger.info("Constructed model from configuration (from_config logic).")

    # Calculate the total number of parameters.
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {format_param_count(total_params)} parameters.")

    return model


def prepare_tokenizer(identifier: str) -> AutoTokenizer:
    """
    Loads a tokenizer and configures it with a padding token if it's missing.

    Args:
        identifier (str): The model identifier from the Hugging Face Hub.

    Returns:
        AutoTokenizer: The configured tokenizer instance.
    """
    logger.info(f"Loading tokenizer from '{identifier}'...")
    tokenizer = AutoTokenizer.from_pretrained(identifier, trust_remote_code=True)

    # If the tokenizer doesn't have a pad token, set it to the EOS token.
    # This is a common and safe practice for many models.
    if tokenizer.pad_token is None:
        logger.info("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # For decoder-only models, left-padding is recommended for training and batch inference.
    # We set it here to ensure consistent behavior.
    if tokenizer.padding_side != 'left':
        logger.warning(f"Tokenizer padding_side is '{tokenizer.padding_side}'. Setting to 'left' for consistency.")
        tokenizer.padding_side = "left"

    return tokenizer
