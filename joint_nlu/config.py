# -*- coding: utf-8 -*-
"""
Module for managing NLU model configuration.

This module provides the NLUConfig class, which is used to configure various aspects of the NLU
model, such as model names, label mappings, etc.

Example:
    config = NLUConfig(model_name="bert-base-uncased", num_intents=10, num_slots=15)
    print(config.model_name)
"""
import json
from transformers import TrainingArguments


class NLUConfig:
    """
    Configuration class for setting up a Joint Natural Language Understanding (NLU) model.

    This class encapsulates the configuration settings for initializing a JointNLUModel,
    including the model name, the number of intents and slots for classification and filling,
    and mappings between IDs and labels for both intents and slots.

    Attributes:
        model_name (str): The name of the pre-trained transformer model to use.
        num_intents (int): The number of unique intents that the model should classify.
        num_slots (int): The number of unique slots that the model should fill.
        intent_id2label (dict): Mapping from intent IDs to intent labels.
        intent_label2id (dict): Mapping from intent labels to intent IDs.
        slot_id2label (dict): Mapping from slot IDs to slot labels.
        slot_label2id (dict): Mapping from slot labels to slot IDs.

    Parameters:
        configuration (str): Configuration loaded from json.
        tokenized_dataset (Dataset): Transformers Dataset that was tokenized with tokenizer.
        iob (dict): Dictionary for IOB tagging, including slot label-to-ID mappings and the total
            number of slot types.
    """
    def __init__(self, configuration, tokenized_dataset=None, iob=None):
        self.model_name = configuration['model_name']

        if tokenized_dataset and iob:
            # create intent labels for the model
            intent_labels = tokenized_dataset["train"].features["intent"].names
            self.num_intents = len(intent_labels)
            intent_label2id, intent_id2label = {}, {}
            for i, i_label in enumerate(intent_labels):
                intent_label2id[i_label] = str(i)
                intent_id2label[str(i)] = i_label

            # create slot labels for the model
            self.num_slots = iob.num_classes
            slot_labels = iob.names
            slot_label2id, slot_id2label = {}, {}
            for i, s_label in enumerate(slot_labels):
                slot_label2id[s_label] = str(i)
                slot_id2label[str(i)] = s_label

            self.intent_id2label = intent_id2label
            self.intent_label2id = intent_label2id
            self.slot_id2label = slot_id2label
            self.slot_label2id = slot_label2id
        else:
            self.num_intents = configuration['num_intents']
            self.num_slots = configuration['num_slots']
            self.intent_id2label = configuration['intent_id2label']
            self.intent_label2id = configuration['intent_label2id']
            self.slot_id2label = configuration['slot_id2label']
            self.slot_label2id = configuration['slot_label2id']

        self.trainer_config = configuration.get('trainer', {})
        self.data_collator_config = configuration.get('data_collator', {})

    def get_training_arguments(self):
        """
        Creates a TrainingArguments instance using the trainer configuration.

        Returns:
            TrainingArguments: An instance with training parameters set.
        """
        push_to_hub = self.trainer_config.get('push_to_hub', False)

        return TrainingArguments(
            output_dir=self.trainer_config['repository_id'],
            evaluation_strategy=self.trainer_config['evaluation_strategy'],
            learning_rate=self.trainer_config['learning_rate'],
            per_device_train_batch_size=self.trainer_config.get('per_device_train_batch_size', 16),
            per_device_eval_batch_size=self.trainer_config.get('per_device_eval_batch_size', 16),
            num_train_epochs=self.trainer_config['num_train_epochs'],
            weight_decay=self.trainer_config['weight_decay'],
            push_to_hub=push_to_hub
        )

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json_string(cls, json_string):
        """
        Creates a NLUConfig instance from a JSON string.

        Args:
            json_string (str): A JSON-formatted string representing the configuration.

        Returns:
            NLUConfig: An instance of NLUConfig initialized with the provided JSON string.
        """
        config_dict = json.loads(json_string)
        return cls(config_dict)
