"""
Bag of Words Attack: https://arxiv.org/pdf/2406.17975
"""
import logging

import numpy as np
from datasets import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from attacks import AbstractAttack


class BagofWordsAttack(AbstractAttack):
    """
    Bag of Words membership inference attack (no target model required).

    Uses word occurrence features with a Random Forest classifier.
    This is a model-agnostic baseline that exploits vocabulary differences
    between members and non-members. Uses min_df=0.05 for word filtering.
    """

    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)

    def run(self, dataset: Dataset) -> Dataset:
        """Execute Bag of Words attack using cross-validation."""
        bow_probas = np.zeros(len(dataset))
        n_splits = int(1 / self.config["test_size"])
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config["seed"])

        for train_idx, test_idx in splitter.split(dataset["text"], dataset["label"]):
            X_train = np.array(dataset["text"])[train_idx]
            y_train = np.array(dataset["label"])[train_idx]
            X_test = np.array(dataset["text"])[test_idx]

            vectorizer = CountVectorizer(min_df=self.config['min_df'])
            X_train_bow = vectorizer.fit_transform(X_train).toarray()
            logging.debug(
                f"Using min_df of {self.config['min_df']}, "
                f"comes down to {len(vectorizer.get_feature_names_out())} features."
            )
            X_test_bow = vectorizer.transform(X_test).toarray()

            classifier = RandomForestClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                min_samples_leaf=self.config['min_samples_leaf']
            )
            classifier.fit(X_train_bow, y_train)
            bow_probas[test_idx] = classifier.predict_proba(X_test_bow)[:, 1]

        dataset = dataset.map(lambda x, i: {self.name: bow_probas[i]}, with_indices=True)
        return dataset

    def extract_features(self, batch):
        """Extract TF-IDF features for ensemble methods."""
        texts = batch["text"]
        if not hasattr(self, 'vectorizer'):
            self.vectorizer = TfidfVectorizer(min_df=self.config['min_df'])
            self.vectorizer.fit(texts)

        bow_features = self.vectorizer.transform(texts).toarray()
        return bow_features
