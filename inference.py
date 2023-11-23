#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs inference using a Joint NLU Model from the Hugging Face Hub.

It clones a specified model repository, loads the model and its configuration, and then performs
inference on provided text input. The script accepts the model name (including the Hugging Face
username) and the text for inference as command-line arguments.

Usage:
    python inference.py --text "Your text here" --model "HF_username/model_name"
"""
import os
import argparse
from joint_nlu.model import JointNLUModel
from joint_nlu.config import NLUConfig


def clone_repository(model):
    """
    Clones the repository from Hugging Face Hub to the specified directory if not already present.

    Args:
        model (str): Full model name (including HF username).

    Returns:
        str: Path to the cloned repository.
    """
    model_name = os.path.basename(model)
    if not os.path.exists(model_name):
        os.system(f"git clone https://huggingface.co/{model}")
    else:
        print(f"Repository {model} already cloned in.")
    return model


def load_model_and_config(model_name,
                          model_filename='jointnlu_model.pth',
                          config_filename='config.json'):
    """
    Loads the model and configuration from the specified directory.

    Args:
        model_path (str): Name of the model file to load.

    Returns:
        JointNLUModel: The loaded NLU model.
        NLUConfig: The configuration object for the model.
    """
    model_path = os.path.basename(model_name)
    config_path = os.path.join(model_path, config_filename)

    # Load the configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = NLUConfig.from_json_string(f.read())

    # Load the model
    model_file = os.path.join(model_path, model_filename)
    model = JointNLUModel(config, trained_model_path=model_file)

    return model, config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform inference with a Joint NLU Model.')
    parser.add_argument('--text', type=str, required=True, help='Input text for inference.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (from HF, with username).')

    args = parser.parse_args()

    # Clone the repository
    clone_repository(args.model)

    # Load model and configuration
    model, config = load_model_and_config(args.model)

    # Perform inference
    prediction = model.predict_intent_and_slots(args.text)
    print(f'Prediction: {prediction}')
