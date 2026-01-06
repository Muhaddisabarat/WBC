"""
Ensemble Attack: https://arxiv.org/abs/2506.10424
"""
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from attacks import AbstractAttack
from attacks.misc.utils import load_reference, make_recall_prefix


class DecisionTreeAttack(AbstractAttack):
    """
    Decision Tree ensemble attack for membership inference.

    Trains a decision tree classifier on multiple attack signals:
    - Loss and perplexity features
    - Lowercase normalization
    - Zlib compression ratio
    - Multiple Min-K%++ thresholds (k=0.2, 0.3, 0.4, 0.5, 0.6)
    - Reference model ratio
    - ReCaLL features

    Requires 'reference_model_path' and 'extra_non_member_dataset' in config.
    """

    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.reference_model, self.reference_tokenizer = load_reference(self.config['reference_model_path'])
        self.reference_device = torch.device('cpu')
        self.extra_non_member_dataset = load_dataset(
            config.get('extra_non_member_dataset', 'imperial-cpg/copyright-traps-extra-non-members'),
            split=config.get('split', 'seq_len_100')
        )
        self.n_shots = config.get('n_shots', 7)

        self.classifier = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=12,
            min_samples_leaf=6,
            max_features='log2',
            class_weight={0: 1.5, 1: 1.0},
            criterion='entropy',
            splitter='best',
            random_state=42
        )
        self.max_len = self.config.get('max_length', 512)
        self.is_trained = False

    def _compute_loss_fp32(self, model, input_ids, attention_mask):
        """Compute per-sample loss in float32 precision for numerical stability."""
        with torch.cuda.amp.autocast(enabled=False):
            input_ids = input_ids.to(dtype=torch.long)
            attention_mask = attention_mask.to(dtype=torch.long)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.to(dtype=torch.float32)

            if isinstance(outputs, tuple):
                logits = logits[0]

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

            return loss.detach().cpu().numpy()

    def extract_features(self, batch):
        """Extract multiple attack signals as features for the classifier."""
        inputs = self.tokenizer(
            batch["text"],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_len
        ).to(self.device)

        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, :-1, :].to(dtype=torch.float32)
                labels = inputs['input_ids'][:, 1:].to(dtype=torch.long)

                token_losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction='none'
                ).reshape(labels.size())

                mean_losses = token_losses.mean(dim=1)

                lower_inputs = self.tokenizer(
                    [text.lower() for text in batch["text"]],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.max_len
                ).to(self.device)

                lower_outputs = self.model(**lower_inputs)
                lower_logits = lower_outputs.logits[:, :-1, :].to(dtype=torch.float32)
                lower_labels = lower_inputs['input_ids'][:, 1:].to(dtype=torch.long)
                lower_losses = F.cross_entropy(
                    lower_logits.reshape(-1, lower_logits.size(-1)),
                    lower_labels.reshape(-1),
                    reduction='none'
                ).reshape(lower_labels.size())
                lower_perplexities = torch.exp(lower_losses.mean(dim=1))

                zlib_scores = torch.tensor([
                    len(zlib.compress(text.encode())) / len(text.encode())
                    for text in batch["text"]
                ], device=self.device)

                k_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
                minkpp_features = []
                for threshold in k_thresholds:
                    k = max(1, int(token_losses.size(1) * threshold))
                    sorted_losses, _ = torch.sort(token_losses, dim=1)
                    minkpp_score = sorted_losses[:, :k].mean(dim=1)
                    minkpp_features.append(minkpp_score.unsqueeze(1))

                ref_inputs = self.reference_tokenizer(
                    batch["text"],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.max_len
                ).to(self.device)

                ref_outputs = self.reference_model(**ref_inputs)
                ref_logits = ref_outputs.logits[:, :-1, :].to(dtype=torch.float32)
                ref_labels = ref_inputs['input_ids'][:, 1:].to(dtype=torch.long)
                ref_losses = F.cross_entropy(
                    ref_logits.reshape(-1, ref_logits.size(-1)),
                    ref_labels.reshape(-1),
                    reduction='none'
                ).reshape(ref_labels.size())
                ref_mean_losses = ref_losses.mean(dim=1)
                ratio_scores = -mean_losses / ref_mean_losses

                recall_texts = [self._build_one_prefix() + " " + text for text in batch["text"]]
                recall_inputs = self.tokenizer(
                    recall_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.max_len
                ).to(self.device)

                recall_outputs = self.model(**recall_inputs)
                recall_logits = recall_outputs.logits[:, :-1, :].to(dtype=torch.float32)
                recall_labels = recall_inputs['input_ids'][:, 1:].to(dtype=torch.long)
                recall_losses = F.cross_entropy(
                    recall_logits.reshape(-1, recall_logits.size(-1)),
                    recall_labels.reshape(-1),
                    reduction='none'
                ).reshape(recall_labels.size())
                recall_mean_losses = recall_losses.mean(dim=1)

                batch_features = torch.cat([
                    mean_losses.to(self.device).unsqueeze(1),
                    zlib_scores.unsqueeze(1),
                    lower_perplexities.to(self.device).unsqueeze(1),
                ] + [score.to(self.device) for score in minkpp_features] + [
                    ratio_scores.to(self.device).unsqueeze(1),
                    recall_mean_losses.to(self.device).unsqueeze(1)
                ], dim=1)

                return batch_features.cpu().numpy()


    def _train_classifier(self, train_dataset):
        """Train the decision tree classifier on extracted features."""
        features = []
        labels = []

        for i in range(0, len(train_dataset), self.config['batch_size']):
            batch = train_dataset[i:i + self.config['batch_size']]
            batch_features = self.extract_features(batch)
            features.append(batch_features)
            labels.extend(batch['label'])

        features = np.vstack(features)
        labels = np.array(labels)

        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        features_normalized = (features - feature_means) / (feature_stds + 1e-8)

        X_train, X_val, y_train, y_val = train_test_split(
            features_normalized,
            labels,
            test_size=0.15,
            random_state=42,
            stratify=labels
        )

        self.classifier.fit(X_train, y_train)

        self.feature_means = feature_means
        self.feature_stds = feature_stds

        val_score = self.classifier.score(X_val, y_val)
        print(f"Validation accuracy: {val_score:.4f}")

        self.is_trained = True

    def _score(self, batch):
        """Compute membership scores using the trained classifier."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before scoring")

        features = self.extract_features(batch)
        features_normalized = (features - self.feature_means) / (self.feature_stds + 1e-8)

        probas = self.classifier.predict_proba(features_normalized)
        return {self.name: probas[:, 1].tolist()}

    def run(self, dataset: Dataset) -> Dataset:
        """Execute decision tree attack on the dataset."""
        if not self.is_trained:
            self._train_classifier(dataset)

        dataset = dataset.map(
            lambda x: self._score(x),
            batched=True,
            batch_size=self.config['batch_size']
        )
        return dataset

    def _build_one_prefix(self, perplexity_bucket=None):
        """Build a prefix from the auxiliary non-member dataset."""
        return make_recall_prefix(
            dataset=self.extra_non_member_dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=perplexity_bucket
        )
