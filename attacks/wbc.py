import logging
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from attacks import AbstractAttack
from attacks.misc.utils import load_reference, unload_reference, compute_per_token_losses


class WbcAttack(AbstractAttack):
    """
    Window-Based Comparison (WBC) Attack for membership inference.

    Compares per-token losses between target and reference models using
    sliding window analysis to detect training data membership.

    Supports both single window and multi-window configurations:
    - Single window: Use `context_window_length` parameter
    - Multi-window: Use `context_window_lengths` list parameter (recommended)

    When using multiple windows, scores are aggregated using mean.
    Requires 'reference_model_path' in config.
    """

    def __init__(self, name: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 config: Dict[str, Any], device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        # Window configuration - support both single and multiple windows
        self._setup_window_config()

        # Reference model setup
        ref_device_name = self.config.get('reference_device', str(self.device))
        self.ref_device = self.device if ref_device_name.lower() == 'same' else torch.device(ref_device_name)

        if 'reference_model_path' not in self.config:
            raise ValueError("WbcAttack requires 'reference_model_path' in its configuration.")

        ref_model_path = self.config['reference_model_path']
        ref_tokenizer_path = self.config.get('reference_tokenizer_path', ref_model_path)

        self.ref_model, self.ref_tokenizer = load_reference(
            model_path=ref_model_path,
            tokenizer_path=ref_tokenizer_path,
            device=self.ref_device,
            hf_token=self.config.get('hf_token')
        )

        # Log configuration
        if len(self.context_window_lengths) > 1:
            logging.info(
                f"WbcAttack configured with reference model: {ref_model_path}, "
                f"window_sizes: {self.context_window_lengths}"
            )
        else:
            logging.info(
                f"WbcAttack configured with reference model: {ref_model_path}, "
                f"window_size: {self.context_window_lengths[0]}"
            )

    def _setup_window_config(self) -> None:
        """Setup window configuration from config parameters."""
        # Check for multiple windows first (preferred)
        if 'context_window_lengths' in self.config:
            lengths = self.config['context_window_lengths']
            if not isinstance(lengths, list) or not lengths:
                raise ValueError("'context_window_lengths' must be a non-empty list.")
            self.context_window_lengths = lengths
        elif 'context_window_length' in self.config:
            # Single window (backward compatibility)
            length = self.config['context_window_length']
            if length <= 0:
                raise ValueError("context_window_length must be positive.")
            self.context_window_lengths = [length]
        else:
            # Default to single window of size 1
            self.context_window_lengths = [1]

    def _compute_window_score(self, target_losses: np.ndarray, ref_losses: np.ndarray,
                              window_size: int) -> float:
        """
        Compare losses within sliding windows using binary scoring.

        When window_size=1, this is equivalent to token-by-token comparison.
        If min_length < window_size, uses min_length as effective window_size.
        """
        min_length = min(len(target_losses), len(ref_losses))

        if min_length == 0:
            return 0.0

        # Use min_length as window_size if text is shorter than desired window
        effective_window_size = min(window_size, min_length)

        # Trim both arrays to same length and compute sliding window sums
        target_trimmed = target_losses[:min_length]
        ref_trimmed = ref_losses[:min_length]

        kernel = np.ones(effective_window_size)
        target_sums = np.convolve(target_trimmed, kernel, mode='valid')
        ref_sums = np.convolve(ref_trimmed, kernel, mode='valid')

        # Binary scoring: fraction of windows where reference loss > target loss
        return np.mean(ref_sums > target_sums)

    def _process_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[float]]:
        """Process a batch of texts and compute WBC scores."""
        texts = batch['text']
        max_len = self.config.get('max_length', 512)

        # Get token losses from both models
        target_losses, target_mask = compute_per_token_losses(
            self.get_base_model(), self.tokenizer, texts, self.device, max_len
        )
        ref_losses, ref_mask = compute_per_token_losses(
            self.ref_model, self.ref_tokenizer, texts, self.ref_device, max_len
        )

        batch_scores = []

        for i in range(len(texts)):
            target_valid = target_losses[i][target_mask[i]]
            ref_valid = ref_losses[i][ref_mask[i]]

            # Compute score for each window size
            window_scores = [
                self._compute_window_score(target_valid, ref_valid, window_size)
                for window_size in self.context_window_lengths
            ]

            # Aggregate using mean
            score = np.mean(window_scores) if window_scores else 0.0
            batch_scores.append(score)

        return {self.name: batch_scores}

    def run(self, dataset: Dataset) -> Dataset:
        """Run WBC attack on the dataset."""
        logging.info(f"Starting WbcAttack: {self.name}")
        batch_size = self.config.get('batch_size', 1)

        try:
            # Create fingerprint based on window configuration
            processed_dataset = dataset.map(
                self._process_batch,
                batched=True,
                batch_size=batch_size,
            )
        except Exception as e:
            logging.error(f"Error during WbcAttack: {e}")
            raise
        finally:
            unload_reference(self.ref_model, self.ref_tokenizer, self.ref_device)
            self.ref_model, self.ref_tokenizer = None, None

        logging.info(f"WbcAttack {self.name} finished.")
        return processed_dataset
