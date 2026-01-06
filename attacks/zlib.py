"""
Zlib Attack: https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
"""
import zlib

from attacks import AbstractAttack
from datasets import Dataset


def _compute_zlib_score(record, ep=1e-5):
    """
    Compute zlib-normalized membership score.

    Normalizes model loss by text compressibility to account for
    intrinsic complexity differences between samples.
    """
    text = record["text"]
    loss = record["nlloss"]
    zlib_entropy = len(zlib.compress(text.encode())) / (len(text) + ep)
    return -loss / (zlib_entropy + ep)


class ZlibAttack(AbstractAttack):
    """
    Zlib compression-based membership inference attack.

    Normalizes the model's loss by the zlib compression ratio of the text.
    This accounts for the fact that some texts are inherently more predictable
    (compressible) than others. Members should have lower loss relative to
    their compressibility.
    """

    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)

    def run(self, dataset: Dataset) -> Dataset:
        """Compute membership scores using loss/compression ratio."""
        dataset = dataset.map(lambda x: {self.name: _compute_zlib_score(x)})
        return dataset