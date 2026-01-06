"""
Membership Inference Attacks (MIA) Framework.

This module provides a framework for implementing and running membership inference
attacks against language models. Each attack implements the AbstractAttack base class
which defines the common interface for all attacks.

Supported attacks include loss-based, ratio-based, and reference model comparison methods.
"""
import hashlib
import json

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


class AbstractAttack(ABC):
    """
    Abstract base class for membership inference attacks.

    All attack implementations must inherit from this class and implement
    the run() method. The base class handles model wrapping (DataParallel)
    and provides common utilities like signature generation and feature extraction.

    Attributes:
        model: Target language model to attack.
        tokenizer: Tokenizer associated with the target model.
        config: Attack-specific configuration dictionary.
        name: Unique identifier for the attack.
        device: Torch device for computation.
    """

    @abstractmethod
    def __init__(self, name: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 config: Dict[str, Any], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.name = name
        self.device = device

        if isinstance(model, torch.nn.DataParallel):
            self.device = next(model.parameters()).device
        else:
            self.device = model.device

    @abstractmethod
    def run(self, dataset: Dataset) -> Dataset:
        """
        Execute the attack on the provided dataset.

        Args:
            dataset: HuggingFace Dataset containing 'text' and 'label' columns.

        Returns:
            Dataset with an additional column containing attack scores.
        """
        pass

    def signature(self, dataset: Dataset) -> str:
        """
        Generate a unique signature for caching attack results.

        Args:
            dataset: The dataset being attacked.

        Returns:
            32-character hexadecimal hash string.
        """
        config_str = json.dumps(self.config, sort_keys=True)
        encoded = (str(dataset.split) + self.name + config_str).encode()
        hash_obj = hashlib.sha256(encoded)
        return hash_obj.hexdigest()[:32]

    def get_base_model(self) -> PreTrainedModel:
        """Unwrap DataParallel model if necessary."""
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def extract_features(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a batch for ensemble methods.

        Args:
            batch: Dictionary containing batch data with attack scores.

        Returns:
            Feature array of shape (batch_size, n_features).
        """
        scores = np.array([-s for s in batch[self.name]]).reshape(-1, 1)
        return scores
