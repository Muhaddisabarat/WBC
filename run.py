"""
Membership Inference Attack Runner.

Main entry point for executing membership inference attacks against language models.
Supports multiple attack types, datasets, and model configurations via YAML config files.

Usage:
    python run.py -c config.yaml --output ./results --target-model gpt2
"""
import argparse
import logging
import os
import time
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from tabulate import tabulate
from transformers import PreTrainedTokenizer, PreTrainedModel

from attacks.misc.utils import batch_nlloss
from utils import (
    get_available_attacks, load_attack, load_config,
    set_seed, save_metadata, is_huggingface_repo_id, init_model, get_printable_ds_name,
    results_with_bootstrapping, generate_metadata_filename, load_mimir_dataset, load_json_dataset,
    get_dataset_type
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

global_config: Optional[Dict[str, Any]] = None


def init_dataset(
        ds_info: Dict[str, Any],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        batch_size: int,
        test_samples: Optional[int] = None,
):
    """Load and preprocess a dataset, calculating NLLoss based on model properties."""
    global global_config
    dataset_type = get_dataset_type(ds_info)
    if dataset_type == "json":
        dataset = load_json_dataset(ds_info["json_train_path"], ds_info["json_test_path"])
    elif dataset_type == "mimir":
        dataset = load_mimir_dataset(name=ds_info["mimir_name"], split=ds_info["split"])
    elif dataset_type == "hf":
        dataset = load_dataset(ds_info["name"], name=ds_info.get("config_name"), split=ds_info["split"],
                               trust_remote_code=True)
    if test_samples is not None and 0 < test_samples < len(dataset):
        dataset = dataset.shuffle(seed=global_config.get("seed", 42)).select(range(test_samples))

    def process_batch(batch):
        nlloss_batch_result = batch_nlloss(
            batch,
            model,
            tokenizer,
            device,
            max_length=global_config["max_length"],
        )
        return {**batch, **nlloss_batch_result}

    dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=[col for col in dataset.column_names if col not in ['label', 'text', 'nlloss']],
        load_from_cache_file=False,
        num_proc=global_config.get("dataset_map_num_proc", 1),
        new_fingerprint=f"dataset_{get_printable_ds_name(ds_info)}"
    )
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run Membership Inference Attacks on Diffusion LLMs.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument('--output', type=str, required=True, help="Directory to save results and metadata.")
    parser.add_argument('--target-model', type=str, help="Path to the base model. Overrides config if provided.")
    parser.add_argument('--target-tokenizer', type=str, help="Path to the tokenizer. Overrides config if provided.")
    parser.add_argument('--reference-model', type=str,
                        help="Path to the reference model. Overrides config if provided.")
    parser.add_argument('--lora-path', type=str, help="Optional path to LoRa adapter. Overrides config if provided.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--base-dir', type=str, default="./",
                        help='Base directory for resolving relative paths in config.')
    parser.add_argument('--dataset-overrides', action='append', default=[],
                        help="Override dataset parameters, format: index:key=value")

    return parser.parse_args()


def main():
    args = parse_args()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    config = load_config(args.config)
    global global_config
    global_config = config['global']
    global_config['seed'] = args.seed

    device = torch.device(global_config["device"])

    load_from_base_dir = global_config.get("load_from_base_dir", False)

    def resolve_path(path_val, base_dir_val):
        if not path_val:
            return path_val
        if is_huggingface_repo_id(path_val):
            return path_val
        if load_from_base_dir and not os.path.isabs(path_val):
            return os.path.join(base_dir_val, path_val)
        return path_val

    model_path_from_config = global_config.get('target_model')
    model_path = args.target_model if args.target_model is not None else model_path_from_config
    if not model_path:
        raise ValueError("Target model path must be provided via --target-model or in config 'global.target_model'.")
    model_path = resolve_path(model_path, args.base_dir)

    tokenizer_path_from_config = global_config.get('tokenizer', model_path)
    tokenizer_path = args.target_tokenizer if args.target_tokenizer is not None else tokenizer_path_from_config
    tokenizer_path = resolve_path(tokenizer_path, args.base_dir)
    lora_path_from_config = global_config.get('lora_adapter_path')
    lora_path_to_load = args.lora_path if args.lora_path is not None else lora_path_from_config
    lora_path_to_load = resolve_path(lora_path_to_load, args.base_dir)
    logging.info(f"Loading base model from: {model_path}")
    if lora_path_to_load:
        logging.info(f"Attempting to load LoRa adapter from: {lora_path_to_load}")
    model, tokenizer, device = init_model(model_path, tokenizer_path, device, lora_adapter_path=lora_path_to_load)

    # Handle reference model override
    ref_model_from_config = global_config.get('reference_model_path')
    ref_model_to_load = args.reference_model if args.reference_model is not None else ref_model_from_config
    ref_model_to_load = resolve_path(ref_model_to_load, args.base_dir)
    if ref_model_to_load:
        global_config['reference_model_path'] = ref_model_to_load  # Update global config for all attacks
        logging.info(f"Using reference model from: {ref_model_to_load}")

    # Parse dataset overrides
    dataset_overrides = {}
    for override in args.dataset_overrides:
        try:
            index_str, key_value = override.split(':', 1)
            index = int(index_str)
            key, value = key_value.split('=', 1)
            if index not in dataset_overrides:
                dataset_overrides[index] = {}
            dataset_overrides[index][key] = value
        except ValueError:
            logging.warning(f"Invalid override format: {override}")

    header = ["MIA Attack", "AUC"] + [f"TPR@FPR={t}" for t in global_config["fpr_thresholds"]]
    results_to_print = {}
    effective_model_description = model_path
    if lora_path_to_load:
        effective_model_description += f" (LoRa: {os.path.basename(lora_path_to_load)})"

    metadata = {
        "timestamp": current_time,
        "model": effective_model_description,
        "config_file": args.config,
        "config_content": config,
        "results": {}
    }

    for idx, ds_info in enumerate(global_config['datasets']):
        if load_from_base_dir:
            for key in ["json_train_path", "json_test_path"]:
                if key in ds_info and not os.path.isabs(ds_info[key]):
                    ds_info[key] = os.path.join(args.base_dir, ds_info[key])

        if idx in dataset_overrides:
            override = dataset_overrides[idx]
            original_type = get_dataset_type(ds_info)
            # Check for full override to a different dataset type
            if ("json_train_path" in override or "json_test_path" in override) and original_type != "json":
                ds_info = {
                    "json_train_path": override.get("json_train_path", ds_info.get("json_train_path")),
                    "json_test_path": override.get("json_test_path", ds_info.get("json_test_path"))
                }
            elif "mimir_name" in override and original_type != "mimir":
                ds_info = {
                    "mimir_name": override["mimir_name"],
                    "split": override.get("split", ds_info.get("split", "train"))
                }
            elif "name" in override and original_type != "hf":
                ds_info = {
                    "name": override["name"],
                    "config_name": override.get("config_name", ds_info.get("config_name")),
                    "split": override.get("split", ds_info.get("split", "train"))
                }
            else:
                # Partial override
                ds_info.update(override)

        logging.info(f"Initializing dataset: {get_printable_ds_name(ds_info)}")
        dataset = init_dataset(
            ds_info=ds_info,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=global_config["batch_size"],
            test_samples=global_config.get("test_samples"),
        )
        ds_name = get_printable_ds_name(ds_info)
        ground_truth_labels = dataset['label']
        attack_results_for_ds = []
        predicted_scores_for_ds = {}

        for attack_name, actual_attack_module_name in get_available_attacks(config).items():
            logging.info(f"Running attack '{attack_name}' on dataset '{ds_name}'")
            attack_config = config.get(attack_name, {}).copy()
            attack_config.update(global_config)
            attack_config['base_dir'] = args.base_dir

            set_seed(args.seed)
            attack_instance = load_attack(attack_name, actual_attack_module_name, model, tokenizer, attack_config,
                                          device)
            processed_dataset_with_attack_scores = attack_instance.run(dataset)

            if 'label' not in processed_dataset_with_attack_scores.column_names:
                raise ValueError(f"Dataset missing 'label' column after running attack '{attack_name}'")
            if attack_name not in processed_dataset_with_attack_scores.column_names:
                raise ValueError(f"Attack scores column '{attack_name}' not found in dataset after running attack.")

            y_true = processed_dataset_with_attack_scores['label']
            y_score = processed_dataset_with_attack_scores[attack_name]

            predicted_scores_for_ds[attack_name] = list(map(float, y_score))
            current_attack_metrics = results_with_bootstrapping(
                y_true, y_score,
                fpr_thresholds=global_config["fpr_thresholds"],
                n_bootstraps=global_config["n_bootstrap_samples"]
            )
            attack_row_data = [attack_name] + current_attack_metrics
            attack_results_for_ds.append(attack_row_data)
            logging.info(f"Attack '{attack_name}' results on '{ds_name}': AUC & TPRs {current_attack_metrics}")

        table_for_ds = tabulate(attack_results_for_ds, headers=header, tablefmt="outline")
        results_to_print[ds_name] = table_for_ds
        metadata["results"][ds_name] = {
            "attacks": [row[0] for row in attack_results_for_ds],
            "results_table_rows": attack_results_for_ds,
            "results_header": header,
            "ground_truth_labels": list(map(int, ground_truth_labels)),
            "predicted_scores": predicted_scores_for_ds
        }

    metadata_filename_str = generate_metadata_filename(
        current_time, model_path, global_config['datasets'],
        global_config["batch_size"], global_config.get("test_samples"),
        global_config["fpr_thresholds"], args.config, args.seed,
        lora_adapter_path=lora_path_to_load
    )

    save_metadata(metadata, args.output, metadata_filename_str)

    for ds_name, table_str in results_to_print.items():
        print(f"\nResults for Dataset: {ds_name}")
        print(table_str)
    print(f"\nMetadata saved to: {os.path.join(args.output, metadata_filename_str)}")
    print(f"Model: {effective_model_description}")


if __name__ == '__main__':
    main()
