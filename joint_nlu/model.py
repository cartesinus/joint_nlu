# -*- coding: utf-8 -*-
"""
JointNLUModel(nn.Module): A PyTorch module for the Joint NLU task which uses a pretrained
transformer model as its encoder, and includes separate classification layers for intents and
slots.
"""
from functools import lru_cache
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from .config import NLUConfig


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
    def __init__(self, config: NLUConfig, trained_model_path=None):
        super().__init__()
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.intent_classifier = nn.Linear(self.encoder.config.hidden_size, config.num_intents)
        self.slot_classifier = nn.Linear(self.encoder.config.hidden_size, config.num_slots)

        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        return {"intent_logits": intent_logits, "slot_logits": slot_logits}

    def predict_intent_and_slots(self, utterance):
        """
        Predicts the intent and slots for a given input utterance.

        Args:
            utterance (str):

        Returns:
            tuple: A tuple containing:
                - intent_label (str): The predicted intent label.
                - word_level_slots (list of str): A list of predicted slot labels for each word.
        """
        tokenizer = self.get_tokenizer()
        inputs = tokenizer(utterance, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)

        intent_label = self.convert_intent_logits_to_labels(outputs['intent_logits'])
        slot_labels = self.convert_slot_logits_to_labels(outputs['slot_logits'])
        word_level_slots = self.align_slots_to_words(slot_labels, offset_mapping)

        return {
            'intent': intent_label,
            'slots': word_level_slots
        }

    @lru_cache()
    def get_tokenizer(self):
        """
        Lazily loads and returns the tokenizer associated with the model.

        This method initializes the tokenizer only once using the model's configuration and caches
        it for future use.

        Returns:
            AutoTokenizer: The tokenizer loaded from the model's pretrained name.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self.tokenizer

    def convert_intent_logits_to_labels(self, intent_logits):
        """
        Converts intent logits to a human-readable intent label.

        Args:
            intent_logits (torch.Tensor): Logits tensor from the intent classifier.

        Returns:
            str: The predicted intent label.
        """
        predicted_intent_id = torch.argmax(intent_logits, dim=1).item()
        predicted_intent_label = self.config.intent_id2label[str(predicted_intent_id)]
        return predicted_intent_label

    def convert_slot_logits_to_labels(self, slot_logits):
        """
        Converts slot logits for each token in the sequence to human-readable slot labels.

        Args:
            slot_logits (torch.Tensor): Logits tensor from the slot classifier.

        Returns:
            list of str: A list of predicted slot labels for each token.
        """
        predicted_slot_ids = torch.argmax(slot_logits, dim=2)
        predicted_slot_labels = [
            self.config.slot_id2label[str(slot_id.item())] for slot_id in predicted_slot_ids[0]
        ]

        return predicted_slot_labels

    def load_trained_model(self, model_path):
        """
        Loads a trained model state dictionary from the specified file path.

        Args:
            model_path (str): Path to the saved model state dictionary.

        This method updates the current model instance with the state loaded from the file.
        """
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.load_state_dict(model_state_dict)

    def align_slots_to_words(self, slot_labels, offset_mapping):
        """
        Aligns sub-token slot labels to word-level in a tokenized sentence.

        Args:
            slot_labels (list of str): Slot labels for each sub-token.
            offset_mapping (torch.Tensor): Start and end indices of sub-tokens.

        Returns:
            list of str: Word-level slot labels.
        """
        word_level_slots = []
        previous_word_end = None
        current_word_slot = None

        for slot_label, (start, end) in zip(slot_labels, offset_mapping.squeeze(0)):
            # Skip special tokens
            if start == end == 0:
                continue

            if start != previous_word_end:
                # We've reached a new word
                if current_word_slot is not None:
                    word_level_slots.append(current_word_slot)
                current_word_slot = slot_label
            else:
                # We're in the same word
                if 'i-' + current_word_slot.split('-')[-1] == slot_label:
                    # If the current slot label is a continuation of the previous slot label
                    # (e.g., previous: 'b-date', current: 'i-date'), we keep the current one
                    continue
                # If the current slot label is different, update it (handle cases: 'fri' and 'day')
                current_word_slot = slot_label

            previous_word_end = end

        # Add the slot label for the last word
        if current_word_slot is not None:
            word_level_slots.append(current_word_slot)

        return word_level_slots
