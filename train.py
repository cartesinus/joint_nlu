#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for training Joint NLU model.
"""
import os
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
from huggingface_hub import notebook_login, whoami, HfFolder, Repository


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
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face authentication token (optional).')
    args = parser.parse_args()

    # Use the token from the command line or environment variable
    token = args.hf_token or os.environ.get('HF_TOKEN')

    # Set the Hugging Face token if provided
    if token:
        HfFolder.save_token(token)    # Use the token from the command line or environment variable

    configuration = read_config(args.config)

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
    push_to_hub = nlu_config.trainer_config.get('push_to_hub', False)

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
        logging.info("Results on testset:")
        logging.info(trainer.evaluate(tokenized_datasets["test"]))

        # Assuming push_to_hub=True in config and from huggingface_hub import notebook_login was
        # called
        if push_to_hub:
            trainer.push_to_hub()

            user_info = whoami(token)
            username = user_info['name']

            # Get the repository name from the configuration
            repository_name = configuration.get('repository_id')

            # Construct the repository URL
            repo_url = f'https://huggingface.co/{username}/{repository_name}'
            repo_dir = "local_repo"
            os.makedirs(repo_dir, exist_ok=True)

            repo = Repository(repo_dir, clone_from=repo_url, use_auth_token=True)

            # Add the nlu_config.json file to the cloned repository
            config_json = nlu_config.to_json_string()
            with open(os.path.join(repo_dir, 'nlu_config.json'), 'w') as f:
                f.write(config_json)

            # Commit and push the changes
            repo.git_add('nlu_config.json')
            repo.git_commit("Add NLU config")
            repo.git_push()

            logging.info("Model pushed to Hugging Face Hub.")

    except Exception as e:
        logging.error("Error: An error occurred while training the model: %s", e)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sys.exit(1)
