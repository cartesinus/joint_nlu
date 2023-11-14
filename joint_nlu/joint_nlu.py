# -*- coding: utf-8 -*-
"""
This module defines a set of classes for training a Joint Natural Language Understanding (NLU) model
using the Hugging Face Transformers library. The Joint NLU model aims to simultaneously perform
intent classification and slot filling tasks in a single forward pass.

Classes:
    JointNLUModel(nn.Module): A PyTorch module for the Joint NLU task which uses a pretrained
    transformer model as its encoder, and includes separate classification layers for intents and
    slots.

    DataCollatorForJointIntentAndSlotFilling(DataCollatorWithPadding): A data collator that inherits
    from Hugging Face's DataCollatorWithPadding. It prepares batch data for training or evaluating
    the Joint NLU model by padding the input sequences and processing intent and slot label data
    accordingly.

    CustomTrainer(Trainer): An extension of Hugging Face's Trainer class, customized to compute
    losses for both intent classification and slot filling tasks. It includes methods to handle the
    custom data collator and implements hooks for additional functionality during the training
    process.

The module is designed to be used with the Hugging Face training loop, providing a streamlined
process for training a Joint NLU model with custom behavior for loss computation and data handling.

Usage:
    The classes defined in this module can be used to set up a training pipeline for a Joint NLU
    model. After initializing the `JointNLUModel` with the appropriate model and task specifics, the
    `CustomTrainer` can be used to train and evaluate the model, utilizing the
    `DataCollatorForJointIntentAndSlotFilling` for efficient data loading and batching.

Examples:
    >>> model = JointNLUModel('bert-base-uncased', num_intents=10, num_slots=20, ...)
    >>> data_collator = DataCollatorForJointIntentAndSlotFilling(tokenizer)
    >>> trainer = CustomTrainer(model=model, args=training_args, data_collator=data_collator, ...)
    >>> trainer.train()
"""
import json
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, DataCollatorWithPadding, Trainer, TrainingArguments


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
    def __init__(self, configuration, tokenized_dataset, iob):
        self.model_name = configuration['model_id']

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

        self.trainer_config = configuration.get('trainer', {})
        self.data_collator_config = configuration.get('data_collator', {})

    def get_training_arguments(self):
        """
        Creates a TrainingArguments instance using the trainer configuration.

        Returns:
            TrainingArguments: An instance with training parameters set.
        """
        return TrainingArguments(
            output_dir=self.trainer_config['repository_id'],
            evaluation_strategy=self.trainer_config['evaluation_strategy'],
            learning_rate=self.trainer_config['learning_rate'],
            per_device_train_batch_size=self.trainer_config.get('per_device_train_batch_size', 16),
            per_device_eval_batch_size=self.trainer_config.get('per_device_eval_batch_size', 16),
            num_train_epochs=self.trainer_config['num_train_epochs'],
            weight_decay=self.trainer_config['weight_decay'],
        )

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.__dict__, indent=2)


class JointNLUModel(nn.Module):
    """
    A joint model for natural language understanding (NLU) that performs both
    intent classification and slot filling in a single forward pass.

    Attributes:
        encoder (AutoModel): A transformer model from Hugging Face's transformers
            library pre-trained on the given model_name.
        config (NLUConfig): Configuration object with model setup parameters, including
            model name, intent and slot counts, and label-ID mappings.
        intent_classifier (nn.Linear): A linear layer for intent classification
            tasks, projecting encoder outputs to intent space.
        slot_classifier (nn.Linear): A linear layer for slot filling tasks,
            projecting encoder outputs to slot space.

    Parameters:
        config: NLUConfig
    """
    def __init__(self, config: NLUConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.intent_classifier = nn.Linear(self.encoder.config.hidden_size, config.num_intents)
        self.slot_classifier = nn.Linear(self.encoder.config.hidden_size, config.num_slots)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs a forward pass on the inputs to produce intent and slot logits.

        Parameters:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding
                token indices.

        Returns:
            dict: A dictionary with the following key-value pairs:
                "intent_logits" (torch.Tensor): Logits for intent classification of shape
                    (batch_size, num_intents).
                "slot_logits" (torch.Tensor): Logits for slot filling of shape
                    (batch_size, sequence_length, num_slots).
        """
        outputs = self.encoder(input_ids, attention_mask=attention_mask)

        intent_logits = self.intent_classifier(outputs.last_hidden_state[:, 0, :])
        slot_logits = self.slot_classifier(outputs.last_hidden_state)

        return {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits
        }


class DataCollatorForJointIntentAndSlotFilling(DataCollatorWithPadding):
    """
    A custom data collator that prepares data batches for joint intent recognition
    and slot filling tasks. This collator extends the `DataCollatorWithPadding` class
    from the Hugging Face transformers library to add additional functionality for
    handling the intent and slot annotations.

    Attributes:
        tokenizer (PreTrainedTokenizer or PreTrainedTokenizerFast): The tokenizer used
            for encoding the text inputs.
        padding (bool): If set to True, the collator will pad the inputs to the maximum
            length in the batch.

    Parameters:
        tokenizer (PreTrainedTokenizer or PreTrainedTokenizerFast): The tokenizer to use
            for preparing the inputs.
        padding (bool, optional): Whether or not to pad the sequences to the maximum length
            in the batch. Defaults to True.
    """
    def __init__(self, tokenizer, padding=True):
        super().__init__(tokenizer=tokenizer, padding=padding)

    def __call__(self, features):
        """
        Processes a list of features into a batch for training or evaluation.

        Each feature in the list is a dictionary with keys corresponding to model inputs
        (`input_ids`, `attention_mask`, `labels`, `intent`, `iob`). This method extracts
        these inputs, applies padding as necessary, and returns a dictionary suitable
        for feeding into a model.

        Parameters:
            features (List[dict]): A list of feature dictionaries, each containing
                the keys `input_ids`, `attention_mask`, `labels`, `intent`, and `iob`.

        Returns:
            dict: A dictionary with keys corresponding to model inputs:
                `input_ids`, `attention_mask`, `labels`, `intent`, and `iob` (padded
                and converted to tensors where appropriate).
        """
        # Only keep relevant fields
        relevant_features = [{k: v for k, v in feat.items()
                             if k in ['input_ids', 'attention_mask', 'labels']}
                                 for feat in features]

        batch = super().__call__(relevant_features)

        batch["intent"] = torch.tensor([f["intent"] for f in features], dtype=torch.long)

        # Pad 'iob' sequences and then stack them into a single tensor
        iob_list = [torch.tensor(f["iob"], dtype=torch.long) for f in features]
        iob_tensor = torch.stack([F.pad(iob, (0, 512 - len(iob)), 'constant', 0)
                                      for iob in iob_list])

        batch["iob"] = iob_tensor

        return batch


class CustomTrainer(Trainer):
    """
    A custom trainer class that extends the Hugging Face Trainer to compute the loss for
    a joint intent classification and slot filling model.

    This trainer handles the training process for models that output both intent and slot
    predictions and computes the loss for each output separately. The losses are then
    combined to create a total loss for the backpropagation step.

    Methods:
        compute_loss: Computes the loss for intent classification and slot filling tasks.
        get_train_dataloader: Returns a DataLoader for the training data with a custom collate
            function.
        on_epoch_end: Custom actions to perform at the end of each epoch.
        get_eval_dataloader: Returns a DataLoader for the evaluation data with a custom collate
            function.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for both intent classification and slot filling tasks.

        Parameters:
            model (nn.Module): The model being trained.
            inputs (dict): A dictionary of inputs to the model. Must include 'intent' and 'iob' for
                labels.
            return_outputs (bool, optional): If set to True, the method also returns model outputs.
                Defaults to False.

        Returns:
            tuple or torch.Tensor: A tuple containing the loss and model outputs if return_outputs
                is True. Otherwise, returns a torch.Tensor representing the loss.
        """
        labels = inputs.pop("intent")
        slot_labels = inputs.pop("iob")

        # intent_logits, slot_logits = model(**inputs)
        outputs = model(**inputs)
        intent_logits = outputs["intent_logits"]
        slot_logits = outputs["slot_logits"]

        intent_loss = F.cross_entropy(intent_logits, labels)
        slot_loss = F.cross_entropy(slot_logits.view(-1, model.config.num_slots),
                                                     slot_labels.view(-1))

        loss = intent_loss + slot_loss  # You can also weigh these losses

        outputs = {"intent_logits": intent_logits, "slot_logits": slot_logits,
                   "intent_labels": labels, "slot_labels": slot_labels
                   }
        return [loss, outputs] if return_outputs else loss

    def get_train_dataloader(self):
        """
        Overrides the Trainer method to return a DataLoader with a custom collate function
        for the training dataset.

        Returns:
            DataLoader: The DataLoader object for the training data.
        """
        # Override to modify collate_fn
        return DataLoader(
            self.train_dataset,
            collate_fn=DataCollatorForJointIntentAndSlotFilling(tokenizer=self.tokenizer),
            batch_size=self.args.train_batch_size,
            shuffle=True,
            drop_last=True  # Skip the last incomplete batch
        )

    def get_eval_dataloader(self, eval_dataset):
        """
        Overrides the Trainer method to return a DataLoader with a custom collate function
        for the evaluation dataset.

        Parameters:
            eval_dataset: The dataset to evaluate on.

        Returns:
            DataLoader: The DataLoader object for the evaluation data.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        # Override to modify collate_fn
        return DataLoader(
            eval_dataset,
            collate_fn=DataCollatorForJointIntentAndSlotFilling(tokenizer=self.tokenizer),
            batch_size=self.args.eval_batch_size,
        )
