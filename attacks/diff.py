"""
Ratio Attack: https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
"""
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from attacks import AbstractAttack
from attacks.misc.utils import load_reference


class DiffAttack(AbstractAttack):
    """
    Difference-based membership inference attack.

    Computes the difference between reference model loss and target model loss.
    Unlike the Ratio attack which uses division, this uses subtraction for
    a more direct comparison. Members show higher difference scores.

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
        self.max_len = self.config.get('max_length', 512)
        self.reference_device = torch.device('cpu')

    def _compute_loss_fp32(self, model, input_ids, attention_mask):
        """Compute per-sample loss in float32 precision for numerical stability."""
        with torch.cuda.amp.autocast(enabled=False):
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
                loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)

                result = loss.detach().cpu().numpy()

                del outputs, logits, shift_logits, shift_targets, shift_attention_mask, loss
                torch.cuda.empty_cache()

                return result

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                raise e

    def _diff_score(self, batch):
        """Compute difference scores: reference_loss - target_loss."""
        target_tokenized = self.tokenizer.batch_encode_plus(
            batch["text"],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_len
        )
        reference_tokenized = self.ref_tokenizer.batch_encode_plus(
            batch["text"],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_len
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

        scores = [r_loss - t_loss for t_loss, r_loss in zip(target_losses, reference_losses)]

        del target_tokenized, reference_tokenized
        torch.cuda.empty_cache()

        return {self.name: scores}

    def extract_features(self, batch):
        """Extract loss-based features for ensemble methods."""
        target_tokenized = self.tokenizer.batch_encode_plus(batch["text"], return_tensors='pt', padding=True)
        target_losses = self._compute_loss_fp32(
            self.model,
            target_tokenized['input_ids'].to(self.device),
            target_tokenized['attention_mask'].to(self.device)
        )

        reference_tokenized = self.ref_tokenizer.batch_encode_plus(batch["text"], return_tensors='pt', padding=True)
        reference_losses = self._compute_loss_fp32(
            self.ref_model,
            reference_tokenized['input_ids'].to(self.reference_device),
            reference_tokenized['attention_mask'].to(self.reference_device)
        )

        ratio_feature = [-t_loss - r_loss for t_loss, r_loss in zip(target_losses, reference_losses)]
        return np.array(ratio_feature).reshape(-1, 1)

    def run(self, dataset: Dataset) -> Dataset:
        """Execute difference attack on the dataset."""
        dataset = dataset.map(
            lambda x: self._diff_score(x),
            batched=True,
            batch_size=self.config.get('batch_size', 4),
            new_fingerprint=f"{self.signature(dataset)}_v5",
        )
        return dataset
