#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for training Joint NLU model.
"""
import sys
import json
import logging
import argparse
from functools import partial
import torch
from transformers import AutoTokenizer
from joint_nlu import (
    NLUConfig,
    JointNLUModel,
    DataCollatorForJointIntentAndSlotFilling,
    CustomTrainer
)
from joint_nlu.utils import preprocess_dataset, tokenize_and_process_labels, compute_metrics


def read_config(config_file):
    """Reads a JSON configuration file and returns a dictionary."""
    try:
        with open(config_file, 'r', encoding='UTF-8') as f:
            configuration = json.load(f)
        return configuration
    except FileNotFoundError:
        logging.error("Error: The file %s does not exist.", config_file)
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error("Error: The file %s is not a valid JSON file.", config_file)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Joint NLU Model")
    parser.add_argument('--config', type=str, default='config/xlm_r-joint_nlu-massive-en.json',
                        help='Path to configuration JSON file.')
    args = parser.parse_args()

    configuration = read_config(args.config)
    push_to_hub = configuration.get("push_to_hub", False)

    logging.basicConfig(filename='train.log', level=logging.INFO)
    logging.info('Started training')

    tokenizer = AutoTokenizer.from_pretrained(configuration['model_id'])
    split_ratio = configuration.get('split_ratio')
    filters = configuration.get('filters')
    dataset, iob = preprocess_dataset(configuration['dataset_id'],
                                      configuration['dataset_configs'],
                                      split_ratio,
                                      filters)
    tokenize_with_tokenizer = partial(tokenize_and_process_labels, tokenizer=tokenizer)
    tokenized_datasets = dataset.map(tokenize_with_tokenizer, batched=True)

    nlu_config = NLUConfig(configuration, tokenized_datasets, iob)
    model = JointNLUModel(nlu_config)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(model)
    logging.info("Total Parameters: %s", total_params)
    logging.info("Trainable Parameters: %s", trainable_params)

    # Prepare custom data collator
    data_collator = DataCollatorForJointIntentAndSlotFilling(
        tokenizer=tokenizer,
        padding=nlu_config.data_collator_config['padding'],
        # todo:        max_length=nlu_config.data_collator_config['max_length'],
    )

    # Prepare custom trainer
    training_args = nlu_config.get_training_arguments()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    try:
        trainer.train()
        trainer.evaluate(tokenized_datasets["test"])

        # Assuming push_to_hub=True in config and from huggingface_hub import notebook_login was
        # called
        if push_to_hub:
            trainer.push_to_hub()
            logging.info("Model pushed to Hugging Face Hub.")

    except Exception as e:
        logging.error("Error: An error occurred while training the model: %s", e)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sys.exit(1)
