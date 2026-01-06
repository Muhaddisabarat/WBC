import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedTokenizer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)

def initialize_trainer(
        config: Dict[str, Any],
        model: PreTrainedModel,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        tokenizer: PreTrainedTokenizer,
) -> Trainer:
    """
    Construct the Trainer with TrainingArguments.
    """
    logger_init = logging.getLogger(__name__)
    training_cfg = config["training"]
    lora_cfg = config.get("lora", {})

    run_name = config.get("run_name", "diffusion-llm-run")

    # LoRA setup (unchanged)
    if lora_cfg.get("enabled", False):
        logger_init.info("LoRA is enabled. Preparing model for PEFT...")
        is_quantized = getattr(model, "is_loaded_in_8bit", False) or \
                       getattr(model, "is_loaded_in_4bit", False)
        use_gradient_checkpointing_for_prep = training_cfg.get("gradient_checkpointing", False)

        if not training_cfg.get("deepspeed_config"):
            if use_gradient_checkpointing_for_prep and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        if is_quantized or (use_gradient_checkpointing_for_prep and not training_cfg.get("deepspeed_config")):
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=use_gradient_checkpointing_for_prep
            )
            logger_init.info(
                f"Model prepared for k-bit/gradient checkpointing compatibility (use_gradient_checkpointing_for_prep={use_gradient_checkpointing_for_prep}).")

        lora_r = lora_cfg.get("r", 8)
        lora_alpha = lora_cfg.get("lora_alpha", 16)
        lora_dropout = lora_cfg.get("lora_dropout", 0.05)
        lora_target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        lora_bias = lora_cfg.get("bias", "none")
        raw_task_type = lora_cfg.get("task_type", "CAUSAL_LM")
        peft_task_type = getattr(TaskType, raw_task_type.upper(), None)

        if peft_task_type is None:
            logger_init.warning(f"Invalid LoRA task_type '{raw_task_type}'. Defaulting to CAUSAL_LM.")
            peft_task_type = TaskType.CAUSAL_LM
        elif peft_task_type.value != raw_task_type.upper() and hasattr(TaskType, raw_task_type.upper()):
            logger_init.warning(
                f"LoRA task_type '{raw_task_type}' resolved to '{peft_task_type}'. Ensure this is intended."
            )

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=peft_task_type,
        )
        model = get_peft_model(model, peft_config)
        logger_init.info("Successfully applied LoRA to the model.")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    # Training arguments setup (unchanged)
    training_args_dict = {
        "output_dir": os.path.join(training_cfg["output_dir"], "ckpts"),
        "num_train_epochs": training_cfg.get("num_train_epochs", 5),
        "max_steps": training_cfg.get("max_steps", -1),
        "per_device_train_batch_size": training_cfg["batch_size"],
        "per_device_eval_batch_size": training_cfg.get("eval_batch_size", training_cfg["batch_size"]),
        "eval_accumulation_steps": training_cfg.get("eval_accumulation_steps"),
        "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
        "learning_rate": training_cfg["learning_rate"],
        "weight_decay": training_cfg["weight_decay"],
        "warmup_steps": training_cfg.get("warmup_steps", 0),
        "save_total_limit": training_cfg.get("save_total_limit", -1),
        "fp16": training_cfg.get("fp16", False),
        "bf16": training_cfg.get("bf16", True),
        "eval_strategy": training_cfg.get("eval_strategy", "epoch") if eval_dataset is not None else "no",
        "save_strategy": training_cfg.get("save_strategy", "epoch"),
        "logging_strategy": training_cfg.get("logging_strategy", "steps"),
        "eval_steps": training_cfg.get("eval_steps", None) if (
                eval_dataset is not None and training_cfg.get("eval_strategy") == "steps") else None,
        "save_steps": training_cfg.get("save_steps", None) if training_cfg.get("save_strategy") == "steps" else None,
        "logging_steps": training_cfg.get("logging_steps", 50),
        "load_best_model_at_end": training_cfg.get("load_best_model_at_end", False),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_loss") if training_cfg.get(
            "load_best_model_at_end", False) else None,
        "report_to": ["wandb"] if config.get("wandb_project") else training_cfg.get("report_to", "none"),
        "run_name": run_name,
        "logging_dir": training_cfg.get("logging_dir", os.path.join(training_cfg["output_dir"], "logs")),
        "deepspeed": training_cfg.get("deepspeed_config", None),
        "remove_unused_columns": True,
        "gradient_checkpointing": training_cfg.get("gradient_checkpointing", False) and not training_cfg.get(
            "deepspeed_config"),
    }

    if training_args_dict["load_best_model_at_end"] and not training_args_dict["metric_for_best_model"]:
        training_args_dict["metric_for_best_model"] = "eval_loss"
        logger_init.info("load_best_model_at_end is True, setting metric_for_best_model to 'eval_loss'.")

    training_args = TrainingArguments(**training_args_dict)

    # Callbacks setup (unchanged)
    callbacks = []
    if "early_stopping_patience" in training_cfg:
        if not training_args.load_best_model_at_end:
            training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = "eval_loss"
            logger_init.info("Enabled load_best_model_at_end and metric_for_best_model for early stopping.")

        early_stopping_threshold = training_cfg.get("early_stopping_threshold", 0.0)
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_cfg["early_stopping_patience"],
                early_stopping_threshold=early_stopping_threshold
            )
        )
        logger_init.info(
            "EarlyStopping enabled: patience=%s, threshold=%.4f",
            training_cfg["early_stopping_patience"], early_stopping_threshold
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    logger_init.info(f"Initialized Standard Hugging Face Trainer with run_name={run_name}.")

    if lora_cfg.get("enabled", False) and hasattr(model, 'peft_config'):
        logger_init.info(f"LoRA Config for the model: {model.peft_config if hasattr(model, 'peft_config') else 'N/A'}")

    return trainer


def run_training(trainer: Trainer, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]) -> None:
    """
    Run training and save final model+tokenizer.
    """
    current_logger = logging.getLogger(__name__)
    current_logger.info("Starting training...")
    training_cfg = config["training"]

    resume_from_checkpoint = training_cfg.get("resume_from_checkpoint", None)
    if resume_from_checkpoint is True:
        current_logger.info(f"Attempting to resume training from the latest checkpoint in {trainer.args.output_dir}.")
        train_output = trainer.train(resume_from_checkpoint=True)
    elif isinstance(resume_from_checkpoint, str):
        current_logger.info(f"Attempting to resume training from checkpoint: {resume_from_checkpoint}.")
        train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        train_output = trainer.train()

    current_logger.info("Training complete.")
    current_logger.info(f"TrainOutput: {train_output}")

    final_model_save_path = config["training"]["output_dir"]
    os.makedirs(final_model_save_path, exist_ok=True)

    current_logger.info(f"Saving model to {final_model_save_path}...")
    model_to_save = trainer.model

    # Check if the model is a PeftModel (LoRA)
    if isinstance(model_to_save, PeftModel):
        current_logger.info("LoRA (PEFT) model detected. Saving adapter model (delta weights).")
    else:
        current_logger.info("Saving full model.")

    model_to_save.save_pretrained(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    current_logger.info(f"Model/Adapter and tokenizer saved to {final_model_save_path}.")

    lora_cfg = config.get("lora", {})
    if isinstance(model_to_save, PeftModel) and lora_cfg.get("save_merged_model_at_end", False):
        merged_model_path = os.path.join(final_model_save_path, "merged_model")
        os.makedirs(merged_model_path, exist_ok=True)
        current_logger.info(f"Merging LoRA weights and saving full merged model to {merged_model_path}...")
        try:
            merged_model = model_to_save.merge_and_unload()
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            current_logger.info(f"Merged model saved to {merged_model_path}.")
        except Exception as e:
            current_logger.error(f"Could not merge and save LoRA model: {e}. "
                                 "Ensure you have enough CPU RAM/GPU RAM (depending on where merge happens) "
                                 "and the model supports merging (e.g., not all quantized models merge easily).")