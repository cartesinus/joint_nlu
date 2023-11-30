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
from huggingface_hub import whoami, HfFolder, Repository, create_repo, upload_folder
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils._errors import HfHubHTTPError
from joint_nlu.model import JointNLUModel
from joint_nlu.config import NLUConfig
from joint_nlu.data_collator import DataCollatorForJointIntentAndSlotFilling
from joint_nlu.trainer import CustomTrainer
from joint_nlu.data_preprocessing import preprocess_dataset, tokenize_and_process_labels
from joint_nlu.metrics import compute_metrics


def read_config(config_file):
    """Reads a JSON configuration file and returns a dictionary."""
    try:
        with open(config_file, 'r', encoding='UTF-8') as f:
            model_configuration = json.load(f)
        return model_configuration
    except FileNotFoundError:
        logging.error("Error: The file %s does not exist.", config_file)
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error("Error: The file %s is not a valid JSON file.", config_file)
        sys.exit(1)


def push_model_to_hub(repo_dir, auth_token, model_filename, config_filename):
    """
    Pushes the model and its configuration to the specified Hugging Face Hub repository.

    Args:
        repo_name (str): The name of the repository to push to.
        auth_token (str): Hugging Face authentication token.
        model_filename (str): Name of the model file.
        config_filename (str): Name of the nlu config gile.

    This function handles cloning the repository, saving the model and configuration,
    and pushing these changes back to the repository.
    """
    user_info = whoami(auth_token)
    username = user_info['name']
    repo_url = f'https://huggingface.co/{username}/{repo_dir}'

    try:
        try:
            create_repo(f"{username}/{repo_dir}")
        except HfHubHTTPError as e:
            if e.response.status_code == 409:  # Conflict error, repository already exists
                logging.info(f"Repository {username}/{repo_dir} already exists. Skipping creation.")
            else:
                raise  # Re-raise the exception if it's not a 409 error

        upload_folder(
            folder_path=repo_dir,
            repo_id=f"{username}/{repo_dir}",
            repo_type="model",
            multi_commits=True,
            multi_commits_verbose=True,
        )
    except Exception as e:
        logging.error("Error pushing model to Hugging Face Hub: %s", e)
        raise


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

    tokenizer = AutoTokenizer.from_pretrained(configuration['model_name'])
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

        # Save model and config
        CONFIG_FILENAME = 'config.json'
        MODEL_FILENAME = 'jointnlu_model.pth'
        REPO_DIR = configuration.get('repository_id')

        os.makedirs(REPO_DIR, exist_ok=True)
        torch.save(model, os.path.join(REPO_DIR, MODEL_FILENAME))
        config_json = nlu_config.to_json_string()
        with open(os.path.join(REPO_DIR, CONFIG_FILENAME), 'w', encoding='utf-8') as f:
            f.write(config_json)

        # Assuming push_to_hub=True in config and called huggingface_hub import notebook_login
        if push_to_hub:
            push_model_to_hub(REPO_DIR, token, MODEL_FILENAME, CONFIG_FILENAME)

    except Exception as e:
        logging.error("Error: An error occurred while training the model: %s", e)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sys.exit(1)
