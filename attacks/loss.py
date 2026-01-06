"""
Loss Attack: https://arxiv.org/pdf/1709.01604
"""
from attacks import AbstractAttack
from datasets import Dataset


class LossAttack(AbstractAttack):
    """
    Loss-based membership inference attack.

    Uses the negative log-likelihood loss of the target model as a membership
    signal. Lower loss indicates higher likelihood of membership since the
    model has likely seen the sample during training.
    """

    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)

    def run(self, dataset: Dataset) -> Dataset:
        """
        Compute membership scores using negative loss values.

        Expects the dataset to already contain 'nlloss' column from preprocessing.
        Higher scores (less negative loss) indicate higher membership likelihood.
        """
        dataset = dataset.map(lambda x: {self.name: -x["nlloss"]})
        return dataset
