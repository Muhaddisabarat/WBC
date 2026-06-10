"""
Utility functions for the MIA framework.

Provides model initialization, dataset loading, configuration management,
attack loading, and metrics computation with bootstrapping.
"""
import importlib
import json
import logging
import os
import pickle
import random
from random import shuffle
from typing import Any, Dict

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from peft import PeftModel
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_dataset_type(ds_info):
    """Determine dataset type from configuration dictionary."""
    if "json_train_path" in ds_info and "json_test_path" in ds_info:
        return "json"
    elif "mimir_name" in ds_info:
        return "mimir"
    elif "name" in ds_info:
        return "hf"
    else:
        raise ValueError("Unknown dataset type")


def init_model(model_name, tokenizer_name, device, lora_adapter_path=None):
    """
    Initialize model and tokenizer, optionally with LoRA adapter.

    Supports multi-GPU via DataParallel when multiple CUDA devices available.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    logging.info(f"Base model {model_name} loaded. Type: {type(model).__name__}")

    if lora_adapter_path:
        if PeftModel is None:
            raise ImportError("PEFT library is required to load LoRa adapters but it's not installed.")
        logging.info(f"Loading LoRa adapter from: {lora_adapter_path}")
        # Ensure model is on the correct device before loading adapter for some PEFT versions/setups
        model = model.to(device)
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        logging.info("LoRa adapter applied successfully.")

    model = model.to(device)

    if torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):
        logging.info(f"Wrapping model with DataParallel for {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    return model, tokenizer, device


def get_printable_ds_name(ds_info):
    """Generate a printable dataset name from the dataset configuration."""
    name_to_print = "unknown_dataset"
    if "name" in ds_info:
        name_to_print = ds_info["name"]
    elif "mimir_name" in ds_info:
        name_to_print = ds_info["mimir_name"]
    elif "json_train_path" in ds_info:
        parent_dir = os.path.basename(os.path.dirname(ds_info["json_train_path"]))
        name_to_print = parent_dir if parent_dir else "custom_json"

    if "split" in ds_info and ("name" in ds_info or "mimir_name" in ds_info):
        name_to_print = f"{name_to_print}_{ds_info['split']}"
    return name_to_print.replace("/", "_").replace("\\", "_")


def results_with_bootstrapping(y_true, y_pred, fpr_thresholds, n_bootstraps=1000):
    """
    Compute bootstrapped AUC and TPR at given FPR thresholds.
    """
    if not y_true or not y_pred:  # Handle empty inputs
        logging.warning("Empty y_true or y_pred in results_with_bootstrapping. Returning N/A.")
        na_result = "N/A"
        results = [na_result] + [na_result for _ in fpr_thresholds]
        return results

    n = len(y_true)
    if n == 0:  # Should be caught by above, but as a safeguard
        logging.warning("Zero length y_true in results_with_bootstrapping. Returning N/A.")
        na_result = "N/A"
        results = [na_result] + [na_result for _ in fpr_thresholds]
        return results

    aucs = []
    tprs_at_fprs = {fpr_val: [] for fpr_val in fpr_thresholds}

    for _ in range(n_bootstraps):
        if n == 1:  # Handle single sample case for bootstrapping (sample with replacement)
            idx = [0] * n  # Effectively, use the single sample n times
        else:
            idx = np.random.choice(n, n, replace=True)

        y_true_sample = np.array(y_true)[idx]
        y_pred_sample = np.array(y_pred)[idx]

        if len(np.unique(y_true_sample)) < 2:  # Not enough classes to compute ROC
            aucs.append(np.nan)  # Or 0.5, or skip
            for fpr_val in fpr_thresholds:
                tprs_at_fprs[fpr_val].append(np.nan)
            continue

        fpr_bs, tpr_bs, _ = roc_curve(y_true_sample, y_pred_sample)
        aucs.append(auc(fpr_bs, tpr_bs))
        for fpr_val in fpr_thresholds:
            if not fpr_bs.size:  # Handle empty fpr_bs
                tprs_at_fprs[fpr_val].append(np.nan)
                continue
            # Find TPR at the FPR closest to the target fpr_val
            tpr_at_fpr = tpr_bs[np.argmin(np.abs(fpr_bs - fpr_val))]
            tprs_at_fprs[fpr_val].append(tpr_at_fpr)

    # Calculate mean and std, handling NaNs from bootstrapping if any
    mean_auc = np.nanmean(aucs)
    std_auc = np.nanstd(aucs)
    results = [f"{mean_auc:.4f} ± {std_auc:.4f}"]

    for fpr_val in fpr_thresholds:
        mean_tpr = np.nanmean(tprs_at_fprs[fpr_val])
        std_tpr = np.nanstd(tprs_at_fprs[fpr_val])
        results.append(f"{mean_tpr:.4f} ± {std_tpr:.4f}")
    return results


def generate_metadata_filename(current_time, model_path, ds_info_list, batch_size,
                               test_samples, fpr_thresholds, config_name, seed,
                               lora_adapter_path=None):
    """
    Generate a descriptive filename for metadata.
    """
    ds_names = "_".join([get_printable_ds_name(ds) for ds in ds_info_list])
    model_name_base = os.path.basename(model_path)
    if lora_adapter_path:
        lora_name = os.path.basename(lora_adapter_path)
        model_name_base = f"{model_name_base}_lora-{lora_name}"

    config_base_name = os.path.splitext(os.path.basename(config_name))[0]
    test_samples_str = f"{test_samples}" if test_samples is not None else "all"
    fpr_str = "-".join(map(str, fpr_thresholds))

    filename = (f"metadata_{current_time}_model-{model_name_base}_config-{config_base_name}_datasets-{ds_names}"
                f"_bs-{batch_size}_ts-{test_samples_str}_fpr-{fpr_str}_seed-{seed}.pkl")
    return filename


def set_seed(seed):
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)


def load_attack(attack_name, real_attack_name, model, tokenizer, config, device):
    """
    Dynamically load and instantiate an attack class.

    Class name is derived from module name or specified explicitly in config.
    """
    module = importlib.import_module(f"attacks.{config['module']}")

    if 'class_name' in config:
        class_name = config['class_name']
    else:
        class_name = ''.join(word.capitalize().split("-")[0] for word in real_attack_name.split('_')) + 'Attack'
        class_name = class_name.replace('OfWords', 'ofWords')

    logging.info(f"Loading attack: {class_name}")

    attr = getattr(module, class_name)
    ret = attr(
        name=attack_name,
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
    return ret


def get_available_attacks(config):
    """Extract attack configurations from config, preserving order."""
    return {k: v['module'] for k, v in config.items() if k != 'global'}


def load_mimir_dataset(name: str, split: str) -> Dataset:
    """
    Load MIMIR benchmark dataset.

    Handles both label-formatted and member/nonmember formatted datasets.
    """
    dataset = load_dataset("iamgroot42/mimir", name, split=split)

    if 'label' not in dataset.column_names:
        if 'member' in dataset.column_names and 'nonmember' in dataset.column_names:
            all_texts = [dataset['member'][k] for k in range(len(dataset))]
            all_labels = [1] * len(dataset)
            all_texts += [dataset['nonmember'][k] for k in range(len(dataset))]
            all_labels += [0] * len(dataset)

            new_dataset = Dataset.from_dict({"text": all_texts, "label": all_labels})
            return new_dataset
        else:
            raise ValueError(
                "Dataset does not contain 'label' column and cannot be inferred from 'member'/'nonmember' columns")

    return dataset


def _extract_texts(obj):
    """
    Extract a list of text strings from a loaded JSON object.

    Supports two formats:
      - Dataset.to_dict() dump: {"data": {"text": [...]}, "indices": [...]}
      - list of records:        [{"text": ...}, ...]
    """
    if isinstance(obj, dict):
        # Dataset.to_dict()-style dump with nested "data" column dict.
        if "data" in obj and isinstance(obj["data"], dict) and "text" in obj["data"]:
            return list(obj["data"]["text"])
        # Flat column dict: {"text": [...]}
        if "text" in obj and isinstance(obj["text"], list):
            return list(obj["text"])
        raise ValueError(f"Unsupported JSON dict structure; keys={list(obj.keys())}")
    if isinstance(obj, list):
        return [item["text"] for item in obj]
    raise ValueError(f"Unsupported JSON top-level type: {type(obj).__name__}")


def load_json_dataset(train_path, test_path):
    """
    Load dataset from JSON files.

    Train samples are labeled as members (1), test samples as non-members (0).
    Results are shuffled for balanced evaluation.
    """
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = _extract_texts(json.load(f))
    train_labels = [1] * len(train_data)

    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = _extract_texts(json.load(f))
    test_labels = [0] * len(test_data)

    all_texts = train_data + test_data
    all_labels = train_labels + test_labels

    combined = list(zip(all_texts, all_labels))
    shuffle(combined)
    all_texts, all_labels = zip(*combined)

    return Dataset.from_dict({"text": list(all_texts), "label": list(all_labels)})


def save_metadata(metadata, output_dir, filename="metadata.pkl"):
    """Save metadata dictionary to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)
    logging.info(f"Metadata saved to {path}")


def load_metadata(filepath):
    """Load metadata dictionary from a pickle file."""
    with open(filepath, 'rb') as f:
        metadata = pickle.load(f)
    logging.info(f"Metadata loaded from {filepath}")
    return metadata


def is_huggingface_repo_id(path_candidate: str) -> bool:
    """
    Checks if the given string is likely a Hugging Face repository identifier.
    """
    if not path_candidate:
        return False
    if os.path.isabs(path_candidate):
        return False
    if path_candidate.startswith(("./", ".\\", "../", "..\\")):
        return False
    if os.path.sep == '\\' and '\\' in path_candidate:
        return False

    try:
        validate_repo_id(path_candidate)
        return True
    except HFValidationError:
        return False
