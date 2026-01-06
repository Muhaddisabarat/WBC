"""
Ratio Attack: https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from attacks import AbstractAttack
from attacks.misc.utils import load_reference


class RatioAttack(AbstractAttack):
    """
    Reference model ratio attack for membership inference.

    Computes the ratio of target model loss to reference model loss.
    The reference model provides a baseline for expected loss on unseen data.
    Members are expected to have lower target loss relative to reference loss.

    Requires 'reference_model_path' in config.
    """

    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)

        self.ref_model, self.ref_tokenizer = load_reference(
            model_path=self.config['reference_model_path'],
            tokenizer_path=self.config['reference_model_path'],
            device=self.config['device'],
            hf_token=self.config.get('hf_token')
        )
        self.reference_device = torch.device('cpu')
        self.epsilon = self.config.get('epsilon', 1e-8)

    def _compute_loss_fp32(self, model, input_ids, attention_mask):
        """Compute per-sample loss in float32 precision for numerical stability."""
        with torch.amp.autocast('cuda', enabled=False):
            try:
                input_ids = input_ids.to(dtype=torch.long, device=model.device)
                attention_mask = attention_mask.to(dtype=torch.long, device=model.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.to(dtype=torch.float32)

                if isinstance(model, torch.nn.DataParallel):
                    vocab_size = model.module.config.vocab_size
                else:
                    vocab_size = model.config.vocab_size

                shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
                shift_targets = input_ids[..., 1:].contiguous()
                shift_attention_mask = attention_mask[..., :-1]

                shift_targets[shift_attention_mask == 0] = -100

                loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
                loss = loss.view(input_ids.shape[0], -1)

                valid_tokens = torch.clamp(shift_attention_mask.sum(axis=1), min=1)
                loss = loss.sum(axis=1) / valid_tokens

                result = loss.detach().cpu().numpy()

                del outputs, logits, shift_logits, shift_targets, shift_attention_mask, loss
                torch.cuda.empty_cache()

                return result

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                raise e

    def _ratio_score(self, batch):
        """Compute target/reference loss ratio scores."""
        target_tokenized = self.tokenizer.batch_encode_plus(
            batch["text"],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        reference_tokenized = self.ref_tokenizer.batch_encode_plus(
            batch["text"],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        target_losses = self._compute_loss_fp32(
            self.model,
            target_tokenized['input_ids'].to(self.device),
            target_tokenized['attention_mask'].to(self.device)
        )
        reference_losses = self._compute_loss_fp32(
            self.ref_model,
            reference_tokenized['input_ids'].to(self.reference_device),
            reference_tokenized['attention_mask'].to(self.reference_device)
        )

        scores = []
        for t_loss, r_loss in zip(target_losses, reference_losses):
            if np.isnan(t_loss) or np.isnan(r_loss) or np.isinf(t_loss) or np.isinf(r_loss):
                score = 0.0
                logging.debug(f"NaN or Inf detected - t_loss: {t_loss}, r_loss: {r_loss}")
            elif abs(r_loss) < self.epsilon:
                score = 0.0 if abs(t_loss) < self.epsilon else -1.0
            else:
                score = -t_loss / (r_loss + self.epsilon)
            scores.append(score)

        del target_tokenized, reference_tokenized
        torch.cuda.empty_cache()

        return {self.name: scores}

    def extract_features(self, batch):
        """Extract loss-based features for ensemble methods."""
        target_tokenized = self.tokenizer.batch_encode_plus(
            batch["text"],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        target_losses = self._compute_loss_fp32(
            self.model,
            target_tokenized['input_ids'].to(self.device),
            target_tokenized['attention_mask'].to(self.device)
        )

        reference_tokenized = self.ref_tokenizer.batch_encode_plus(
            batch["text"],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        reference_losses = self._compute_loss_fp32(
            self.ref_model,
            reference_tokenized['input_ids'].to(self.reference_device),
            reference_tokenized['attention_mask'].to(self.reference_device)
        )

        ratio_features = []
        for t_loss, r_loss in zip(target_losses, reference_losses):
            if np.isnan(t_loss) or np.isnan(r_loss) or np.isinf(t_loss) or np.isinf(r_loss):
                feature = 0.0
            else:
                feature = -t_loss - r_loss
            ratio_features.append(feature)

        return np.array(ratio_features).reshape(-1, 1)

    def run(self, dataset: Dataset) -> Dataset:
        """Execute ratio attack on the dataset."""
        dataset = dataset.map(
            lambda x: self._ratio_score(x),
            batched=True,
            batch_size=self.config.get('batch_size', 4),
            new_fingerprint=f"{self.signature(dataset)}_v7",
        )
        return dataset
