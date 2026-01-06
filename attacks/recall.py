"""
ReCaLL Attack: https://arxiv.org/pdf/2406.15968
"""
import numpy as np
from datasets import Dataset, load_dataset

from attacks import AbstractAttack
from attacks.misc.utils import compute_nlloss, batch_nlloss, make_recall_prefix


class RecallAttack(AbstractAttack):
    """
    ReCaLL (Relative Conditional Log-Likelihood) membership inference attack.

    Measures the ratio of conditional loss (with prefix) to unconditional loss.
    Uses a fixed prefix drawn from an auxiliary non-member distribution to
    provide context. Members show different conditional behavior than non-members.

    Requires 'extra_non_member_dataset' and 'n_shots' in config.
    """

    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.extra_non_member_dataset = load_dataset(config['extra_non_member_dataset'], split=config['split'])
        self.max_len = self.config.get('max_length', 512)

    def _build_fixed_prefixes(self, target_dataset):
        """Build fixed prefixes, optionally matched by perplexity bucket."""
        if self.config["match_perplexity"]:
            perplexity_buckets = set(x["perplexity_bucket"] for x in target_dataset)
            prefixes = {
                ppl: make_recall_prefix(
                    dataset=self.extra_non_member_dataset,
                    n_shots=self.config["n_shots"],
                    perplexity_bucket=ppl
                )
                for ppl in perplexity_buckets
            }
            return prefixes
        else:
            prefix = make_recall_prefix(
                dataset=self.extra_non_member_dataset,
                n_shots=self.config["n_shots"],
                perplexity_bucket=None
            )
            return [prefix]

    def _build_one_prefix(self, perplexity_bucket=None):
        """Build a single prefix from the auxiliary dataset."""
        return make_recall_prefix(
            dataset=self.extra_non_member_dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket
        )

    def run(self, dataset: Dataset) -> Dataset:
        """Execute ReCaLL attack on the dataset."""
        if 'nlloss' not in dataset.column_names:
            dataset = dataset.map(
                lambda batch: batch_nlloss(batch, self.model, self.tokenizer, self.device),
                batched=True,
                batch_size=self.config['batch_size'],
                remove_columns=[col for col in dataset.column_names if col not in ['label', 'text', 'nlloss']],
            )

        if self.config["fixed_prefix"]:
            prefixes = self._build_fixed_prefixes(dataset)
        else:
            prefixes = None

        dataset = dataset.map(
            lambda x: self._recall_nlloss(x, prefixes=prefixes),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v2",
        )
        dataset = dataset.map(lambda x: {self.name: x['recall_nlloss'] / x['nlloss']})
        return dataset

    def _recall_nlloss(self, batch, prefixes=None):
        """Compute loss with prefix context."""
        if prefixes is not None:
            if self.config["match_perplexity"]:
                texts = [prefixes[ppl_bucket] + " " + text for ppl_bucket,
                         text in zip(batch["perplexity_bucket"], batch["text"])]
            else:
                texts = [prefixes[0] + " " + text for text in batch["text"]]
        else:
            if self.config["match_perplexity"]:
                texts = [self._build_one_prefix(ppl_bucket) + " " + text for ppl_bucket,
                         text in zip(batch["perplexity_bucket"], batch["text"])]
            else:
                texts = [self._build_one_prefix() + " " + text for text in batch["text"]]

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
        return {'recall_nlloss': losses}

    def extract_features(self, batch):
        """Extract loss and recall loss features for ensemble methods."""
        loss_feature = [-l for l in batch['nlloss']]

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
        recall_feature = [-loss for loss in recall_losses]

        features = np.column_stack([loss_feature, recall_feature])
        return features
