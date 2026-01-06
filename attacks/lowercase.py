"""
Lowercase Attack: https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
"""
from attacks import AbstractAttack
from attacks.misc.utils import compute_nlloss
from datasets import Dataset


class LowercaseAttack(AbstractAttack):
    """
    Lowercase normalization attack for membership inference.

    Compares the model's loss on original text versus lowercased text.
    The intuition is that training data memorization is case-sensitive,
    so members will show a larger relative difference in loss when
    case information is removed.
    """

    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.ep = 1e-5

    def run(self, dataset: Dataset) -> Dataset:
        """Compute membership scores using original/lowercase loss ratio."""
        dataset = dataset.map(
            lambda x: self._lowercase_nlloss(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )
        dataset = dataset.map(lambda x: {self.name: -x['nlloss'] / (x['lowercase_nlloss'] + self.ep)})
        return dataset

    def _lowercase_nlloss(self, batch):
        """Compute loss on lowercased versions of input texts."""
        texts = [x.lower() for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['max_length']
        )
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        losses = compute_nlloss(self.model, token_ids, attention_mask)
        return {'lowercase_nlloss': losses}

