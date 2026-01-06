"""
CON-ReCall Attack: https://arxiv.org/pdf/2409.03363
"""
import copy
import logging
import random

import datasets
import numpy as np
from datasets import Dataset, load_dataset

from attacks import AbstractAttack
from attacks.misc.utils import compute_nlloss

logging.basicConfig(level=logging.WARNING)


def _make_conrecall_prefix(dataset, n_shots, perplexity_bucket=None, target_index=None):
    """
    Build a prefix by sampling n_shots examples from the dataset.

    Args:
        dataset: Source dataset for prefix samples.
        n_shots: Number of shots to include in prefix.
        perplexity_bucket: Optional filter by perplexity bucket.
        target_index: Index to exclude (for member prefixes).

    Returns:
        Concatenated prefix string.
    """
    if target_index is not None:
        indices_to_keep = [i for i in range(len(dataset)) if i != target_index]
        dataset = dataset.select(indices_to_keep)

    if perplexity_bucket is not None:
        datasets.disable_progress_bar()
        dataset = dataset.filter(lambda x: x["perplexity_bucket"] == perplexity_bucket)
        datasets.enable_progress_bar()

    all_indices = list(range(len(dataset)))
    n_shots = min(n_shots, len(all_indices))

    if n_shots > len(all_indices):
        logging.warning(
            f"Requested n_shots ({n_shots}) > available population ({len(all_indices)}). "
            f"Reducing to {len(all_indices)}."
        )

    indices = random.sample(all_indices, n_shots)
    prefixes = [dataset[i]["text"] for i in indices]

    return " ".join(prefixes)


class ConrecallAttack(AbstractAttack):
    """
    Contrastive ReCaLL (CON-ReCall) membership inference attack.

    Extends ReCaLL by contrasting member-conditioned and non-member-conditioned
    likelihoods. Uses 10-shot prefixes drawn randomly for each target example.
    Non-member prefixes from auxiliary distribution, member prefixes from
    training data (assuming partial access).
    """

    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.extra_non_member_dataset = load_dataset(config['extra_non_member_dataset'], split=config['split'])
        self.max_len = self.config.get('max_length', 512)

    def _build_non_member_prefix(self, perplexity_bucket=None):
        """Build prefix from non-member auxiliary dataset."""
        return _make_conrecall_prefix(
            dataset=self.extra_non_member_dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket
        )

    def _build_member_prefix(self, target_index, dataset, perplexity_bucket=None):
        """Build prefix from member dataset, excluding target sample."""
        return _make_conrecall_prefix(
            dataset=dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket,
            target_index=target_index
        )

    def _build_one_prefix(self, perplexity_bucket=None):
        """Build a single prefix for feature extraction."""
        return _make_conrecall_prefix(
            dataset=self.extra_non_member_dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket
        )

    def run(self, dataset: Dataset) -> Dataset:
        """Execute CON-ReCall attack computing contrastive conditional likelihoods."""
        ds_clone = copy.deepcopy(dataset)
        dataset = dataset.map(
            lambda x: self._conrecall_nlloss(x, ds_clone),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v7",
        )
        dataset = dataset.map(
            lambda x: {self.name: (x[f'{self.name}_nm_nlloss'] - x[f'{self.name}_m_nlloss']) / x['nlloss']}
        )
        return dataset

    def _conrecall_nlloss(self, batch, dataset):
        """Compute losses with member and non-member conditioned prefixes."""
        if self.config["match_perplexity"]:
            it = enumerate(zip(batch["perplexity_bucket"], batch["text"]))
            non_member_texts = [
                self._build_non_member_prefix(ppl_bucket) + " " + text
                for i, (ppl_bucket, text) in it
            ]

            it = enumerate(zip(batch["perplexity_bucket"], batch["text"]))
            ds_members_only = dataset.filter(lambda x: x["label"] == 1)
            member_texts = [
                self._build_member_prefix(
                    perplexity_bucket=ppl_bucket,
                    target_index=i,
                    dataset=ds_members_only
                ) + " " + text
                for i, (ppl_bucket, text) in it
            ]
        else:
            non_member_texts = [self._build_non_member_prefix() + " " + text for text in batch["text"]]

            ds_members_only = dataset.filter(lambda x: x["label"] == 1)
            member_texts = [
                self._build_member_prefix(
                    target_index=i,
                    dataset=ds_members_only
                ) + " " + text
                for i, text in enumerate(batch["text"])
            ]

        ret = {}
        for texts, label in [(non_member_texts, "nm"), (member_texts, "m")]:
            tokenized = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_len
            )
            token_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            losses = compute_nlloss(self.model, token_ids, attention_mask)
            ret[f"{self.name}_{label}_nlloss"] = losses
        return ret

    def extract_features(self, batch):
        """Extract contrastive recall features for ensemble methods."""
        texts = [self._build_one_prefix() + " " + text for text in batch["text"]]
        recall_tokenized = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_len
        )
        recall_losses = compute_nlloss(
            self.model,
            recall_tokenized['input_ids'].to(self.device),
            recall_tokenized['attention_mask'].to(self.device)
        )
        basic_recall_feature = [-loss for loss in recall_losses]

        contrast_features = []
        for _ in range(self.config.get('n_contrasts', 3)):
            texts = [self._build_one_prefix() + " " + text for text in batch["text"]]
            tokenized = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_len
            )
            losses = compute_nlloss(
                self.model,
                tokenized['input_ids'].to(self.device),
                tokenized['attention_mask'].to(self.device)
            )
            contrast_features.append([-loss for loss in losses])

        features = np.column_stack([basic_recall_feature] + contrast_features)
        return features
