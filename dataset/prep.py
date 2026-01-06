"""
Dataset Preparation Utilities.

Directly run it with `python prep.py` to get all dataset we used in the paper
"""
import json
import multiprocessing
import os

from datasets import load_dataset
from transformers import AutoTokenizer


def create_subset_dataset(
        dataset_name,
        dataset_config_name=None,
        dataset_split="train",
        text_column="text",
        num_sample=100,
        output_dir="subset_dataset",
        context=None,  # optional text context to prepend/append
        member_ratio=0.5,  # fraction of chosen subset that goes into "member"
        min_length=0,  # minimum length of text for an entry (in tokens)
        tokenizer_name="gpt2"  # HuggingFace tokenizer name
):
    """
    Create a membership vs. non-membership subset from a given dataset.
    Entries shorter than `min_length` (token length of the final text)
    will be discarded before splitting.

    Args:
        dataset_name (str):
            Name or path of the dataset to load (HF Hub or local).
        dataset_config_name (str, optional):
            Config or subset name for the dataset (e.g., "wikitext-103-v1").
        dataset_split (str, optional):
            Which split to load ("train", "test", etc.). Defaults to "train".
        text_column (str, optional):
            Which column to read as the main text.
        num_sample (int):
            How many total samples to select for this subset (after length filtering).
        output_dir (str):
            Directory to store the resulting JSON files.
        context (str, optional):
            Optional context string to prepend (or append) to each text.
        member_ratio (float):
            Fraction of samples allocated to the "member" subset. Defaults to 0.5.
        min_length (int, optional):
            Minimum token length for an entry's text content.
            Entries shorter than this will not be included. Defaults to 0 (no minimum length).
        tokenizer_name (str, optional):
            HuggingFace tokenizer name to use for token counting. Defaults to "gpt2".
    """

    # Initialize tokenizer
    print(f"[INFO] Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        # Add pad token if not present (some tokenizers don't have one)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"[INFO] Tokenizer loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer '{tokenizer_name}': {e}")
        raise

    # 1. Load the dataset (with optional config)
    if dataset_config_name:
        dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split, trust_remote_code=True)
    print(f"[INFO] Loaded dataset '{dataset_name}' "
          f"config='{dataset_config_name}' split='{dataset_split}' with {len(dataset)} rows.")

    # 2. Filter by min_length (tokens) before shuffling
    filtered_dataset = dataset
    if min_length > 0:
        print(f"[INFO] Initial number of samples before length filtering: {len(dataset)}")

        # Pre-calculate context tokens once if context exists
        context_tokens = 0
        if context:
            context_tokens = len(tokenizer.encode(context, add_special_tokens=False))

        # Token-based filter function
        def is_long_enough(example):
            try:
                text_content = example[text_column]
                if context:
                    full_text = f"{context} {text_content}"
                else:
                    full_text = text_content

                # Tokenize and count tokens
                tokens = tokenizer.encode(full_text, add_special_tokens=False)
                return len(tokens) >= min_length
            except KeyError:
                print(f"[WARNING] text_column '{text_column}' not found in example")
                return False
            except Exception as e:
                print(f"[WARNING] Error tokenizing text: {e}")
                return False

        original_count = len(dataset)
        filtered_dataset = dataset.filter(
            is_long_enough,
            num_proc=multiprocessing.cpu_count()
        )
        filtered_count = len(filtered_dataset)
        print(
            f"[INFO] Filtered dataset from {original_count} to {filtered_count} rows based on min_length={min_length} tokens.")
        if filtered_count == 0 and original_count > 0:
            print(
                f"[WARNING] All samples were filtered out by min_length={min_length} tokens. No data will be generated.")

    # 3. Shuffle the filtered dataset (faster since it's smaller)
    print(f"[INFO] Shuffling filtered dataset ({len(filtered_dataset)} samples)...")
    shuffled_dataset = filtered_dataset.shuffle(seed=42)

    # 4. Select a random subset
    num_sample_to_select = min(num_sample, len(shuffled_dataset))
    if len(shuffled_dataset) == 0:  # If filtering resulted in an empty dataset
        print(f"[INFO] No samples available after filtering. Output files will be empty.")
        subset_dataset = shuffled_dataset  # empty dataset
    elif num_sample_to_select == 0 and len(shuffled_dataset) > 0:
        print(
            f"[WARNING] num_sample is 0 (or less than 0), but {len(shuffled_dataset)} samples are available after filtering. "
            f"No samples will be selected. Output files will be empty.")
        subset_dataset = shuffled_dataset.select([])  # Create an empty selection with same schema
    else:
        print(f"[INFO] Selecting {num_sample_to_select} samples from the (potentially filtered) dataset ...")
        subset_dataset = shuffled_dataset.select(range(num_sample_to_select))

    # 5. Split into "member" and "non-member" subsets
    member_size = int(member_ratio * len(subset_dataset))
    member_dataset = subset_dataset.select(range(member_size))
    non_member_dataset = subset_dataset.select(range(member_size, len(subset_dataset)))

    # 6. Create data entries for the two subsets
    member_data = []
    non_member_data = []

    # Process member dataset
    for row in member_dataset:
        text_value = row[text_column]
        if context:
            text_value = f"{context} {text_value}"
        member_data.append({"text": text_value})

    # Process non-member dataset
    for row in non_member_dataset:
        text_value = row[text_column]
        if context:
            text_value = f"{context} {text_value}"
        non_member_data.append({"text": text_value})

    # 7. Write out results as a single JSON array per file.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    member_file = os.path.join(output_dir, "train.json")
    non_member_file = os.path.join(output_dir, "test.json")

    print(f"[INFO] Writing member set ({len(member_data)} entries) => {member_file}")
    with open(member_file, "w", encoding="utf-8") as f:
        json.dump(member_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Writing non-member set ({len(non_member_data)} entries) => {non_member_file}")
    with open(non_member_file, "w", encoding="utf-8") as f:
        json.dump(non_member_data, f, ensure_ascii=False, indent=2)

    # 8. Print a sample entry from member/non-member to confirm correctness.
    if member_data:
        print("[INFO] Sample member entry:", member_data[0])
        # Show token count for the sample
        sample_tokens = len(tokenizer.encode(member_data[0]["text"], add_special_tokens=False))
        print(f"[INFO] Sample member entry token count: {sample_tokens}")
    else:
        print("[INFO] No member data to sample.")

    if non_member_data:
        print("[INFO] Sample non-member entry:", non_member_data[0])
        # Show token count for the sample
        sample_tokens = len(tokenizer.encode(non_member_data[0]["text"], add_special_tokens=False))
        print(f"[INFO] Sample non-member entry token count: {sample_tokens}")
    else:
        print("[INFO] No non-member data to sample.")

    print("[INFO] Subset dataset generation complete!")
    print(f"Dataset Name         : {dataset_name}")
    print(f"Config Name          : {dataset_config_name}")
    print(f"Split                : {dataset_split}")
    print(f"Text Column          : {text_column}")
    print(f"Min Length (tokens)  : {min_length if min_length > 0 else 'N/A'}")
    print(f"Tokenizer Used       : {tokenizer_name}")
    print(f"Context Provided     : {context is not None}")
    print(f"Member set size      : {len(member_data)}")
    print(f"Non-member set size  : {len(non_member_data)}")
    print(f"Output directory     : {output_dir}")
    print("---------------------------------------------------\n")


def batch_generate_subsets(dataset_list, num_sample=20000, context=None, member_ratio=0.5, min_length=0,
                           tokenizer_name="gpt2"):
    """
    Given a list of dataset specifications, run create_subset_dataset on each.

    Args:
        dataset_list (list): List of dataset specifications
        num_sample (int): Number of samples per dataset
        context (str, optional): Context to prepend to each text
        member_ratio (float): Fraction allocated to member subset
        min_length (int): Minimum token length for filtering
        tokenizer_name (str): HuggingFace tokenizer name
    """
    for ds_spec in dataset_list:
        ds_name = ds_spec["name"]
        ds_config = ds_spec.get("config", None)
        ds_split = ds_spec.get("split", "train")
        ds_column = ds_spec["text_column"]  # required

        if ds_config:
            out_dir = f"{ds_name.split('/')[-1]}-{ds_config}-subset"
        else:
            out_dir = f"{ds_name.split('/')[-1]}-subset"

        create_subset_dataset(
            dataset_name=ds_name,
            dataset_config_name=ds_config,
            dataset_split=ds_split,
            text_column=ds_column,
            num_sample=num_sample,
            output_dir=out_dir,
            context=context,
            member_ratio=member_ratio,
            min_length=min_length,
            tokenizer_name=tokenizer_name
        )


if __name__ == "__main__":
    MAIN_DATASETS = [
        {
            "name": "EleutherAI/wikitext_document_level",
            "config": "wikitext-103-v1",
            "split": "train",
            "text_column": "page"
        },
        {
            "name": "EdinburghNLP/xsum",
            "config": None,
            "split": "train",
            "text_column": "document"
        },
        {
            "name": "sentence-transformers/reddit",
            "config": None,
            "split": "train",
            "text_column": "body"
        },
        {
            "name": "sentence-transformers/amazon-reviews",
            "config": None,
            "split": "train",
            "text_column": "review"
        },
        {
            "name": "sentence-transformers/ccnews",
            "config": None,
            "split": "train",
            "text_column": "article"
        },
        {
            "name": "HuggingFaceTB/cosmopedia",
            "config": "auto_math_text",
            "split": "train",
            "text_column": "text"
        },
        {
            "name": "HuggingFaceTB/cosmopedia",
            "config": "khanacademy",
            "split": "train",
            "text_column": "text"
        },
        {
            "name": "HuggingFaceTB/cosmopedia",
            "config": "stanford",
            "split": "train",
            "text_column": "text"
        },
        {
            "name": "HuggingFaceTB/cosmopedia",
            "config": "stories",
            "split": "train",
            "text_column": "text"
        },
        {
            "name": "HuggingFaceTB/cosmopedia",
            "config": "web_samples_v2",
            "split": "train",
            "text_column": "text"
        },
        {
            "name": "HuggingFaceTB/cosmopedia",
            "config": "wikihow",
            "split": "train",
            "text_column": "text"
        },
        {
            "name": "HuggingFaceTB/cosmopedia",
            "config": "stanford",
            "split": "train",
            "text_column": "text"
        },
    ]

    # We want to sample 20,000 examples from each,
    batch_generate_subsets(
        dataset_list=MAIN_DATASETS,
        num_sample=20000,
        min_length=512,
        tokenizer_name="EleutherAI/pythia-2.8b"
    )