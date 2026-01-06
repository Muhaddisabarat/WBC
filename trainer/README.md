# Trainer Module

This module provides model fine-tuning capabilities for membership inference attack experiments, including support for various defense mechanisms.

## Module Structure

```
trainer/
├── configs/                     # Configuration generators and templates
│   ├── prep.py                 # Base configuration generator
│   └── ds_config.json          # DeepSpeed configuration
├── misc/                        # Utility modules
│   ├── data.py                 # Dataset loading and preprocessing
│   ├── env_setup.py            # Environment and distributed setup
│   ├── models.py               # Model initialization utilities
│   └── utils.py                # General utilities
├── get_target.py               # Main script for fine-tuning target models
├── train.py                    # Core training logic with defense support
└── setup.py                    # Package setup
```

## Installation

```bash
# From the trainer directory
pip install -e .
```

## Usage

### Fine-tuning Target Models

Train a model on member data for membership inference evaluation:

```bash
python get_target.py \
    --config_path configs/pythia-2.8b-cosmopedia.yaml \
    --base_path ../weights \
    --train_subset_size 10000 \
    --ref_subset_size 10000
```

### Configuration Structure

Example YAML configuration:
```yaml
run_name: pythia-2.8b-cosmopedia-khanacademy
wandb_project: WBC

dataset:
  train:
    type: local
    path: data/train.json
  test:
    type: local
    path: data/test.json

tokenizer:
  identifier: EleutherAI/pythia-2.8b
  max_length: 512

model:
  identifier: EleutherAI/pythia-2.8b
  load_pretrained: true

training:
  output_dir: ./outputs
  batch_size: 16
  gradient_accumulation_steps: 1
  learning_rate: 5.0e-5
  weight_decay: 0.1
  warmup_steps: 500
  num_train_epochs: 3
  eval_strategy: epoch
  save_strategy: epoch
  bf16: true

# Optional: LoRA configuration
lora:
  enabled: true
  r: 32
  lora_alpha: 64
  target_modules: ["query_key_value"]
```

## Command-Line Arguments

### get_target.py

- `--config_path`: Path to YAML configuration file
- `--base_path`: Base directory for saving model weights
- `--train_subset_size`: Number of training samples (-1 for all)
- `--ref_subset_size`: Number of test/validation samples (-1 for all)
- `--local_rank`: Local rank for distributed training
- `--dist_backend`: Backend for distributed training (nccl/gloo)


## Output Structure

Training produces:
```
weights/
└── model-name-config/
    ├── pytorch_model.bin    # Model weights
    ├── config.json          # Model configuration
    ├── tokenizer_config.json
    ├── train_subset.json    # Training data indices
    └── test_subset.json     # Test data indices
```

## Advanced Features

### Multi-GPU Training

Enable DeepSpeed for large models:
```yaml
training:
  deepspeed_config: configs/ds_config.json
```

### Early Stopping

```yaml
training:
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
  load_best_model_at_end: true
```

### Gradient Checkpointing

For memory-efficient training:
```yaml
training:
  gradient_checkpointing: true
```

## Notes

- Differential privacy is incompatible with DeepSpeed and mixed precision
- LoRA models save only adapter weights by default
- All models are evaluated against the WBC attack after training