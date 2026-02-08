# Window-Based Comparison (WBC) Attack

[![Read Blog](https://img.shields.io/badge/Blog-Read_Post-brightgreen.svg)](https://yuetian.me/blog/2026/wbc/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.02751-b31b1b.svg)](https://arxiv.org/abs/2601.02751)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Implementation of Window-Based Comparison (WBC) - a membership inference attack against fine-tuned Large Language Models using localized window-based analysis. 

## Installation

```bash
# Clone repository
git clone https://github.com/Stry233/WBC
cd wbc-attack

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Step 1: Prepare Dataset

Create balanced member/non-member splits from a HuggingFace dataset:

```bash
python dataset/prep.py \
    --dataset_name "HuggingFaceTB/cosmopedia" \
    --config "khanacademy" \
    --num_samples 20000 \
    --min_length 512 \
    --output_dir "cosmopedia-khanacademy-subset"
```

This creates `train.json` (members) and `test.json` (non-members) in the output directory.

### Step 2: Fine-tune Target Model

Fine-tune a model on the member data:

```bash
python trainer/get_target.py \
    --config_path configs/config_all.yaml \
    --base_path ./weights \
    --train_subset_size 10000 \
    --ref_subset_size 10000
```

To get the yaml file for your setup, please modify and run `\trainer\configs\prep.py` based on instructions in that file.

### Step 3: Run Membership Inference Attack

Execute WBC and baseline attacks:

```bash
python run.py \
    --config configs/config_all.yaml \
    --output results/ \
    --base-dir ./weights \
    --seed 42
```

## Configuration

### Main Configuration (`configs/example.yaml`)

```yaml
global:
  target_model: "./path/to/target"
  reference_model_path: "EleutherAI/pythia-2.8b"
  datasets:
    - json_train_path: "data/train.json"
      json_test_path: "data/test.json"
  batch_size: 1
  max_length: 512
  fpr_thresholds: [0.1, 0.01, 0.001]
  n_bootstrap_samples: 100

# WBC attack settings
Wbc:
  module: "wbc"
  reference_model_path: "EleutherAI/pythia-2.8b"
  context_window_lengths: [2, 3, 4, 6, 9, 13, 18, 25, 32, 40]
```

### Attack Selection

Enable/disable attacks by commenting them in `configs/config_all.yaml`:

```yaml
# Reference-free attacks
loss:
  module: loss
zlib:
  module: zlib
  
# Reference-based attacks  
ratio:
  module: ratio
  reference_model_path: "EleutherAI/pythia-2.8b"
  
# Our method
Wbc:
  module: "wbc"
  # ... configuration
```

## Advanced Usage

### Using Custom Datasets

```bash
python dataset/prep.py \
    --dataset_name "your_dataset" \
    --text_column "text" \
    --split "train" \
    --num_sample 20000 \
    --min_length 512 \
    --tokenizer_name "EleutherAI/pythia-2.8b"
```

### Custom Attack Implementation

To add a new attack, create a file in `attacks/`:

```python
from attacks import AbstractAttack

class YourAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        
    def _process_batch(self, batch):
        # Implement your attack logic
        scores = compute_membership_scores(batch)
        return {self.name: scores}
```

Then add to `configs/config_all.yaml`:
```yaml
your_attack:
  module: your_attack
  # your parameters
```

## Outputs

The attack produces:
- **Metadata file**: `metadata_[timestamp]_[config].pkl` containing:
  - Attack scores for all methods
  - Ground truth labels  
  - AUC and TPR metrics
  - Configuration details

- **Console output**: Results table with AUC and TPR@FPR metrics


## Repository Structure

```
├── attacks/                 # Attack implementations
│   ├── wbc.py              # WBC attack implementation
│   └── misc/
│       └── utils.py        # Loss computation utilities
├── trainer/                # Model fine-tuning
│   ├── get_target.py       # Main training script
│   └── configs/            # Training configurations
├── configs/                # Attack configurations
├── dataset/                # Dataset preparation
├── scripts/                # Automation scripts
├── run.py                  # Main attack runner
└── utils.py               # Shared utilities
```

## Requirements

- **GPU Memory**: 
  - Minimum: 8GB (for Pythia-160M)
  - Recommended: 40GB
  
- **Disk Space**: ~5GB per fine-tuned model

- **Python Packages**: See `requirements.txt`

## Cite our work

TBD.