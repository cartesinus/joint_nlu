# -*- coding: utf-8 -*-
"""
A module providing a data collator for joint intent and slot filling.

This module contains the DataCollatorForJointIntentAndSlotFilling class, which extends
Hugging Face's DataCollatorWithPadding. It is designed to format and pad batch data
suitably for training or evaluating a Joint NLU model, ensuring proper handling of both
intent and slot label data in the process.
"""
import torch
import torch.nn.functional as F
from transformers import DataCollatorWithPadding


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
        iob_tensor = torch.stack([F.pad(iob, (0, 128 - len(iob)), 'constant', 0)
                                      for iob in iob_list])

        batch["iob"] = iob_tensor

        return batch
